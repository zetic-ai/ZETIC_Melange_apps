# SPEC: PronunciationScoring — FINAL (GATE 0 cleared; all fields filled)

## One-line pitch
Language-learning pronunciation coach: the user reads a displayed sentence into
the mic; the app scores per-phoneme / per-word pronunciation goodness and an
overall score — fully on-device (edtech / consumer-AI prospects).

## Model
- Source (HF repo / origin): Peacockery/citrinet-256-phoneme-en (NeMo Citrinet-256
  fine-tuned to ARPABET-41 phoneme CTC on LibriSpeech train-clean-100; base
  nvidia/stt_en_citrinet_256_ls). MIT license.
- Architecture: Citrinet-256 (1D separable convs + squeeze-excite, 9.7M params)
  with the exact NeMo log-mel frontend BAKED INTO the ONNX (raw waveform in).
- Melange model name: ajayshah/PronunciationScoring (registered, READY; SDK
  name WITH the slash. Dashboard header "ZETIC | PronunciationScoring" — the
  "ZETIC |" is a display prefix only, never part of the name.)
- Melange version: 1
- Input tensor: float32[1, 81760] — raw MONO 16 kHz waveform, 5.11 s, values in
  [-1, 1]. NO normalization, NO mel — feed raw samples. (81760 samples -> 512
  mel frames -> 64 CTC frames, all fixed.)
- Output tensor: float32[1, 64, 45], row-major [frame][class] — 64 CTC frames
  (one per 80 ms of audio) x 45 LOG-SOFTMAX scores per frame.
- Served input/output shapes (dashboard, GATE-0 paste-back): input audio
  float32[1,81760]; output logprobs float32[1,64,45] — exactly as exported,
  no reshaping by Melange.
- Post-processing baked into ONNX? Log-softmax yes. CTC decode, forced
  alignment, and GOP scoring: NO — pure Dart (below).
- Classes / labels: labels.txt — ids 0-38 ARPABET phonemes (AA=0 ... ZH=38),
  ids 39-43 unused tokenizer specials (never fire; ignore), id 44 = CTC blank.
  WARNING: NeMo's own tokenizer DISPLAYS id 0 as "[PAD]"; that map is wrong
  (verified empirically) — labels.txt is authoritative.
- modelMode to use and why: RUN_AUTO (orchestrator ruling; no client mode
  reliably steers backend selection — see CLAUDE.md section 5). Benchmark
  reality (GATE-0 paste-back): deployability 88% (FP32 100%, FP16 98%, INT8
  28% — server-side concern only); NPU min 4.73 / med 12.85 ms, CPU med
  69.84 ms, GPU med 104 ms; on Apple devices AUTO benches ~52-77 ms
  (CPU-class) while SPEED benches ~6.6-14 ms (iPhone 15 Pro: auto 52.64 /
  speed 8.78). EXPECT ~50-70 ms per inference under RUN_AUTO on the demo
  iPhone — fine for this app (ONE inference per 5.11 s recording, not
  per-frame). Do not chase modes; surface measured latency + served artifact
  on the HUD. Memory: up to ~273 MB at load/inference (paste-back) — no
  app-side action, just known.

## Input source
- Microphone. Request 16 kHz mono PCM16 (or capture at 44.1/48 kHz and
  downsample in Dart — document which; a clean integer-ratio path is 48 kHz /3).
- Recording UX (binding, measured): record the FULL 5.11 s window (countdown /
  progress ring), so the tail is real room tone. NEVER pad the tensor with
  digital zeros — zero-runs break the in-graph per-utterance normalization and
  measurably wreck accuracy (PER 0.29 -> 0.58). If a capture is somehow short,
  pad with low-level noise (~1e-3 RMS), never zeros.
- Demo sentences must take ~3.5-5 s to read aloud (>=60-70% window fill);
  scoring degrades gracefully but visibly below ~50% fill.
- No orientation concerns (audio).

## Pre-processing pipeline (ordered, exact)
1. Capture exactly 81760 samples @ 16 kHz mono (5.11 s window).
2. PCM16 -> float32: sample / 32768.0 (range [-1, 1]). No other scaling, no
   mean subtraction, no resample beyond the capture-rate conversion.
3. Wrap as Tensor.float32List(data, shape: [1, 81760]).

## Post-processing pipeline (ordered, exact)
Reference implementation (must be reproduced exactly): validation/validate_onnx.py.
1. Read output as Float32List, view as [64][45] row-major (frame-major: index
   = frame*45 + class). Values are log-probabilities.
2. Target phonemes: each demo sentence ships with its precomputed ARPABET id
   sequence (CMUdict, stress digits stripped, first pronunciation) — bundle as
   an asset; no runtime G2P needed for a fixed sentence list.
3. CTC forced alignment (Viterbi) of the target sequence over the 64 frames:
   expand targets as [blank, p1, blank, p2, ..., blank] (blank id 44), standard
   CTC transitions (stay / advance 1 / skip-from-2-back only when the skipped
   state is a blank between DIFFERENT phonemes), backtrack the best path, and
   collect the frame set aligned to each target phoneme.
4. Per-phoneme GOP score = mean over its aligned frames of exp(logprob) of the
   target phoneme id. Phonemes with no aligned frames score 0.
5. Word score = mean of its phonemes' GOP (also surface the min phoneme as the
   "fix this sound" highlight). Overall score = mean over words, mapped through
   worker-calibrated thresholds to a 0-100 or star scale. Calibrate on the
   committed reference clips: correct-text GOP mean ~0.75 at 94% window fill
   (ls1), mismatched-text ~0.14; short-fill correct ~0.19-0.33 vs mismatch
   ~0.01-0.03 — thresholds must account for window fill (blank-frame fraction
   is a usable fill proxy).
6. Optional live extra: greedy CTC decode (argmax per frame, collapse repeats,
   drop blank/specials, map via labels.txt) to display "what we heard" — treat
   as decoration; scoring must use the aligned GOP, which is far more robust.
7. Fluency signal (optional): speech-frame fraction and per-phoneme aligned
   duration vs expectation from the alignment in step 3.

## UI
- Left to the worker. Functional must-haves: display the sentence to read;
  record with visible 5.11 s countdown; per-word (tappable to per-phoneme)
  color-coded scores; overall score; inference latency readout.

## Platform targets
- iOS 16.6+, Android minSdk 24 (match PyroGuard).
- Known OS traps: iOS/macOS 26.3+ CoreML-GPU MPSGraph crash class is handled
  server-side by ZETIC (do not select models/modes around it); read the SERVED
  artifact from the native console. Conv-only graph is the least-risky
  architecture for the NPU compilers.
- Mic permission (NSMicrophoneUsageDescription / RECORD_AUDIO).

## Validation focus (Tier A traps for THIS model)
- Tensor layout: hand-built [1,64,45] tensor with one known max per frame —
  assert frame-major indexing (frame*45 + class), not class-major.
- Label map: assert id 0 -> AA (NOT [PAD]) and blank = 44 (labels.txt is the
  contract; NeMo's display map is a known trap).
- CTC collapse semantics: repeats collapse only across consecutive frames
  WITHOUT an intervening blank ("L <blank> L" -> two L's, "L L" -> one).
- Forced-alignment DP: golden test against validation/validate_onnx.py outputs
  on the committed reference wavs (same aligned frame sets, same GOP scores
  within 1e-3 on the ONNX-runtime side; on-device artifact may differ in
  precision — compare scores with tolerance, not bit-exact).
- Zero-padding trap: unit test that the preprocessor NEVER emits a zero-padded
  tail (assert noise-pad path); this is the single biggest silent-accuracy
  killer found in exploration.
- Threshold behavior: word just-below / just-above the score threshold.
- Sample-rate: assert the capture pipeline really delivers 16 kHz mono before
  tensor fill (a 44.1 kHz buffer fed as 16 kHz shifts every formant).

## GATE 0 — CLEARED
1. Uploaded and READY: ajayshah/PronunciationScoring v1; served shapes echo
   the export exactly (float32[1,81760] -> float32[1,64,45]); modelMode
   RUN_AUTO.
2. Size/quality decision RULED by orchestrator: STAY WITH CITRINET (40 MB /
   18.5% PER). HuBERT-377MB fails the mobile-size rubric; the demo story is
   GOP scoring (proven 5.3-13x correct-vs-mismatch separation), not raw
   transcription. HuBERT remains the documented escalation path ONLY if
   device results disappoint (recorded in HANDOFF.md).
