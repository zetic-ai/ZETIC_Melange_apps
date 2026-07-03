# Model selection — PronunciationScoring (speech / ASR encoder, use-case: pronunciation & fluency scoring)

First exploration of the speech family — the export recipe in `export.py` is the
family's reference recipe (raw-waveform in, DSP frontend baked into the ONNX).

## Shortlist (top 5)

| Rank | HF repo | Downloads | License | Export path | Melange-fit notes | Score |
|------|---------|-----------|---------|-------------|-------------------|-------|
| 1 | Peacockery/citrinet-256-phoneme-en | 21 | MIT | NeMo -> torch wrapper -> ONNX (this repo's recipe) | 9.7M params / **40 MB fp32** — the ONLY mobile-sized phoneme-CTC model found on the Hub. Conv-only (1D separable convs + squeeze-excite): no attention, static shapes trivial, NPU-friendly op set. ARPABET-41 output maps 1:1 onto CMUdict targets for GOP. Weakness: measured PER ~18.5% (vs 11.4% for the 10x-bigger HuBERT) and only 21 downloads / research-grade provenance. | 8.5/10 |
| 2 | Peacockery/hubert-base-phoneme-en | 328 | Apache-2.0 | transformers -> ONNX | Quality winner: measured PER **11.4%** on the same clips, same ARPABET-41 vocab. But 95M params = **377 MB fp32 ONNX** — fails the mobile-size column outright and cannot even be committed to git (>100 MB). | 6/10 |
| 3 | bookbot/wav2vec2-ljspeech-gruut | 2,317 | Apache-2.0 | transformers -> ONNX | Well-regarded IPA-phoneme CTC (built for a kids' reading-tutor app — perfect task pedigree). Same disqualifier: wav2vec2-base, 377 MB fp32. IPA vocab would also need gruut/eSpeak-derived targets instead of CMUdict. | 5.5/10 |
| 4 | slplab/wav2vec2-large-robust-L2-english-phoneme-recognition | 222,891 | none stated | transformers -> ONNX | Best task fit on paper (trained on L2-learner speech, has explicit `*_err` mispronunciation tokens) and hugely downloaded — but wav2vec2-LARGE: **1.26 GB fp32**, and NO license tag (GTM risk). | 4/10 |
| 5 | vitouphy/wav2vec2-xls-r-300m-timit-phoneme | 5,844 | Apache-2.0 | transformers -> ONNX | Popular, documented (IPA, TIMIT), but XLS-R-300M: 1.26 GB fp32. | 3.5/10 |

Also examined: mostafaashahin/wav2vec2-base-timit-phoneme-arpa-39 (377 MB, no
license), ct-vikramanantha/phoneme-scorer-v2-wav2vec2 (377 MB), OthmaneJ/
distil-wav2vec2 (208 MB but CHARACTER-level — poor GOP fit), nvidia/
stt_en_citrinet_256_ls (37 MB but BPE-subword output — unusable for per-phoneme
scoring), QuartzNet15x5 (char-level, NGC-hosted). The Hub search (13 queries,
132 unique models: task=ASR + free text phoneme / pronunciation / espeak /
distil / quartznet / citrinet / mispronunciation, sorted by downloads) found
**no other genuinely mobile-sized phoneme-CTC checkpoint**.

## Winner: Peacockery/citrinet-256-phoneme-en

- The size column is binding: every wav2vec2/HuBERT phoneme model is 377 MB+
  fp32 (the assignment caps "mobile-sized" at ~<100 MB); Citrinet-256 is 40 MB
  and its conv-only graph is the most NPU-compilable architecture in the list.
- Phoneme output (ARPABET-41) is the right currency for GOP scoring — CMUdict
  gives target sequences for any demo sentence offline.
- **Honest quality trade-off, measured head-to-head** (greedy CTC decode vs
  CMUdict reference, 4 real LibriSpeech clips + 2 macOS-`say` TTS clips):
  Citrinet aggregate PER **18.5%** vs HuBERT-base **11.4%**. Citrinet is NOT
  degenerate (no blank-domination; decodes track the reference closely — see
  validation below), but it is a mid-quality acoustic model: expect noisy
  per-phoneme scores and design the app around sentence/word-level
  aggregation. If the human prefers quality over size at GATE 0,
  Peacockery/hubert-base-phoneme-en (same vocab, same scoring head) is the
  drop-in alternative at 377 MB.

## Export

- Recipe: `export.py` (speech-family reference recipe). Baked the exact NeMo
  log-mel frontend (preemph 0.97, hann 400/512 constant-pad STFT as a conv-DFT,
  slaney 80-mel, log+2^-24, per-feature normalization over 511 valid frames)
  into the graph as standard ops, so the app feeds RAW WAVEFORM — no DSP in
  Dart. Verified bit-close against NeMo: frontend max|diff| 5.5e-5, full-graph
  logits max|diff| 9.2e-4, onnxruntime-vs-torch 1.1e-3.
- Input:  float32[1, 81760] — raw mono 16 kHz waveform, 5.11 s, [-1, 1].
  81760 samples -> exactly 512 mel frames (multiple of NeMo pad_to=16) ->
  exactly 64 encoder frames; no padding ambiguity anywhere.
- Output: float32[1, 64, 45] — 64 frames (80 ms hop) x 45 log-softmax classes;
  ids 0-38 = ARPABET (labels.txt), 39-43 unused specials, 44 = CTC blank.
  Post-processing baked in? Log-softmax yes; CTC decode/alignment no (Dart).
- Opset 12 (13/14 also verified working; 12 matches the PyroGuard
  known-good-with-Melange precedent), static shapes confirmed by checker +
  shape inference (no dynamic dims, no Shape/Expand/Where/NonZero/Loop/If —
  the traced mask machinery was constant-folded with polygraphy and the 23
  residual constant-mask `Where` nodes rewritten to Mul/Add; 23 no-op Casts
  stripped; final graph 839 nodes, op set: Add Concat Conv Div Log LogSoftmax
  MatMul Mul Pad Pow ReduceMean ReduceSum Relu Sigmoid Slice Sqrt Sub
  Transpose Unsqueeze).
- onnxsim 0.6.5 SIGBUS-crashes on this graph on macOS-arm64 — recipe uses
  polygraphy fold_constants instead. Recorded so the next speech exploration
  doesn't rediscover it.

## Validation (measured, exported ONNX via onnxruntime, exact app preprocessing)

Harness: `validation/validate_onnx.py` (greedy decode + CTC forced alignment +
per-phoneme GOP = mean aligned target posterior; reference clips committed).

- id-map ground truth: "fox" -> `F <id0> K S`, "dog" -> `D <id0> G` (cot-caught
  merged TTS voice), "shore" -> `SH <id3> R` — so id0=AA, id3=AO, confirming
  vocab.txt order and refuting NeMo's shifted display map (labels.txt is
  authoritative). Specials (39-43) fired 0 times across all clips; blank
  dominates silent tails (fraction 0.88-1.00).
- Realistic demo condition (ls1, real speech, 94% window fill):
  greedy `N AO R IH Z M S T K OW L T ER Z M AE N ER L EH S IH N T ER EH S T IH
  NG DH AH N HH IH Z M AE T ER` vs target — PER 0.163;
  GOP correct-text mean 0.753 vs mismatched-text 0.143 (5.3x separation).
- Worst-case window fill (short TTS clips, 49%/39% fill): decode degrades (PER
  0.52/0.67) but GOP separation survives (0.325 vs 0.031 = 10.6x; 0.187 vs
  0.014 = 13.1x). Scoring (alignment-based) is robust where greedy display is
  not.
- **Trap found and measured: NEVER zero-pad the window.** Digital-zero padding
  drives log-mel frames to log(2^-24) and wrecks the in-graph per-feature
  normalization: ref1 PER 0.29 (noise pad) -> 0.58 (zero pad). The app must
  record the full 5.11 s window (real room tone). Demo sentences should take
  3.5-5 s to read so speech fills >=60-70% of the window.

Honest verdict: a real, working phoneme model with a measured ~18.5% PER
ceiling — good enough for a scoring demo built on aligned-GOP aggregation with
calibrated thresholds; not good enough for a raw-transcript showcase. The
size/quality alternative (HuBERT, 377 MB, 11.4% PER) is documented above for
the GATE-0 decision.
