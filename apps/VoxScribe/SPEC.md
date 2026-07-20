# SPEC: VoxScribe

> Unified spec, merged by the orchestrator from the two Stage-0 Explorer stubs
> (`asr/spec_stub_asr.md`, `diarization/spec_stub_diarization.md`) plus the human's
> GATE-0 paste-back. This is a **3-model pipeline** (Whisper encoder + Whisper decoder
> + pyannote segmentation). GATE-0 fields are now FILLED. No section is TBD.

## One-line pitch
A fully on-device, speaker-labeled live transcript ("who spoke when") demo for prospect
**Kardome** — pyannote segmentation supplies speaker turns; Whisper-tiny transcribes each
turn; the UI paints a who-spoke-when timeline with a visible "on-device, no cloud" signal.
The deliverable is a **screen-recordable demo video before Monday KST**, not a store app.

## Pipeline architecture (binding — this is the integration design, not the worker's to invent)

```
 audio clip (file, ≤10 s, 16 kHz mono float [-1,1])
        │
        ▼
 ┌─────────────────────────┐
 │ 1. SEGMENTATION          │  ajayshah/PyannoteSegmentation v1
 │  x[1,1,160000] → y[1,589,7] (powerset log-softmax)
 └─────────────────────────┘
        │  powerset-decode → per-local-speaker activity → onset/offset + min-duration
        ▼
   speaker segments:  [{start, end, speaker:s}, ...]   ("who spoke when")
        │
        │  FOR EACH speaker segment span (the floor: diarize-THEN-transcribe)
        ▼
 ┌─────────────────────────┐     ┌─────────────────────────┐
 │ 2. WHISPER ENCODER       │ →   │ 3. WHISPER DECODER (×N)  │
 │  span→logmel[1,80,3000]  │     │  greedy 448-step decode  │
 │  → hidden[1,1500,384]    │     │  → token ids → text      │
 └─────────────────────────┘     └─────────────────────────┘
        │
        ▼
   speaker-attributed transcript lines:  [{speaker, start, end, text}, ...]
        │
        ▼
   UI: scrolling transcript with speaker color/label + who-spoke-when timeline + latency/RTF HUD
```

**Why diarize-then-transcribe (the chosen fusion, floor):** the segmentation model defines
the speaker turn *with its timestamps*, so each Whisper transcription is attributed to
exactly one speaker **by construction** — no fragile word-timestamp reconciliation. Whisper
here runs with `<|notimestamps|>` (the reference client seeds only SOT), so it yields text
but no word timing; using diarization to cut spans first sidesteps that entirely. Cost: the
encoder/decoder run once per speaker segment (2–4× for a 2-person ≤10 s clip), which is fine
for a recorded video. **This is the GATE-2 confirm point** — the worker may propose the
alternative "transcribe-once-then-assign-dominant-speaker" (needs Whisper timestamp tokens),
but the floor is diarize-then-transcribe.

---

## Model

### Model 1 — Speaker segmentation
- Source: k2-fsa/sherpa-onnx pre-exported **pyannote/segmentation-3.0**, static-pinned (see `diarization/export.py`).
- Architecture: PyanNet — SincNet conv front end + 2× BiLSTM + linear, **powerset** head (max 3 local speakers, max 2 simultaneous).
- **Melange model name: `ajayshah/PyannoteSegmentation`** — **version 1** ✅ (GATE-0 confirmed, status READY-pending/optimizing)
- Input: `x` float32 `[1,1,160000]` = 10 s mono @ 16 kHz, raw waveform ~[-1,1], **NO** normalization. ✅ served shape confirmed
- Output: `y` float32 `[1,589,7]` = 589 frames × 7 powerset classes, **log-softmax** (`exp(row)`≈1). ✅ served shape confirmed
- Post-processing baked in? **No** — powerset decode + (optional) stitching + segment extraction are pure-Dart.
- Powerset decode table (num_speakers=3, max_simultaneous=2):
  | idx | local speakers | idx | local speakers |
  |-----|----------------|-----|----------------|
  | 0 | {} silence | 4 | {spk0,spk1} |
  | 1 | {spk0} | 5 | {spk0,spk2} |
  | 2 | {spk1} | 6 | {spk1,spk2} |
  | 3 | {spk2} | | |
- modelMode: **RUN_AUTO** ✅. LSTM may force a CPU artifact — acceptable for one 10 s window. Read served `runtimeApType` on the device console.

### Model 2 — Whisper encoder (REUSED, not re-registered)
- Source: openai/whisper-tiny (MIT), encoder half.
- **Melange model name: `OpenAI/whisper-tiny-encoder`** — **version 1** (reused). ✅ READY (85% deployable; AUTO lands on NPU).
- Input: `input_features` float32 `[1,80,3000]` — Whisper log-mel.
- Output (served name `_413`): float32 `[1,1500,384]`. This tensor feeds the decoder's `enc_hidden` input unchanged.

### Model 3 — Whisper decoder (REUSED, not re-registered; no KV-cache, fixed length 448)
- **Melange model name: `OpenAI/whisper-tiny-decoder`** — **version 1** (reused). ✅ READY (81% deployable).
- Inputs, **in this positional order** (the SDK takes `model.run(List<Tensor>)` — order, not name, is what binds): (1) `ids` **int32** `[1,448]`, (2) `enc_hidden` float32 `[1,1500,384]` (= the encoder's `_413` output), (3) `enc_mask` **int32** `[1,448]`.
- Output (served name `_853`): float32 `[1,448,51865]`.
- **Dtype is int32 — confirmed BOTH from shipping-source ground truth** (Android `IntArray`/`Int.SIZE_BYTES`, iOS `Int32`/`MemoryLayout<Int32>.size`) **and from the GATE-0 dashboard read** (`ids int32[1,448]`, `enc_mask int32[1,448]`). Build int32 in Dart.
- Special tokens: SOT 50258, EOT 50257, pad 50256; vocab 51865; `vocab.json` ships from `apps/whisper-tiny`.
- modelMode: **RUN_AUTO** for both encoder and decoder.

---

## Input source
- **Floor: file-based** — a bundled ≤10 s, 2-speaker scripted clip (most reliable to screen-record).
- Stretch: live mic.
- Required format for BOTH model families: **16 kHz, mono, float32 PCM in [-1,1]**. Resample from 44.1/48 kHz and downmix stereo→mono before anything else.
- Orientation handling: N/A (audio). The "on-device" UI signal replaces it as a must-have (airplane-mode-friendly badge + latency/RTF readout).

## Pre-processing pipeline (ordered, exact)
**Shared front end (once per clip):**
1. Acquire PCM → if stereo, downmix to mono (ch0 or average; reference uses ch0).
2. If rate ≠ 16000, resample to 16000.
3. Ensure float32 in [-1,1] (int16 → divide by 32768). NO further normalization.

**For segmentation (Model 1):**
4. Take/zero-pad-to one 10 s window = exactly **160000 samples** → `[1,1,160000]`.

**For Whisper (Models 2+3), per speaker-segment span:**
5. Slice the span's samples; zero-pad/truncate to **480000 samples (30 s)**.
6. Log-mel: STFT `n_fft=400`, `hop=160`, Hann; `n_mels=80`; `log10`; Whisper clamp+scale
   (`log_spec = max(log_spec, log_spec.max()-8.0)` then `(log_spec+4)/4`) → `[1,80,3000]`.
   (Reuse the SDK `WhisperWrapper` mel or a Dart/native equivalent reproducing these exact params.)

## Post-processing pipeline (ordered, exact)
**Segmentation → segments (frame constants: `scale=270/16000=0.016875 s/frame`, `scale_offset=991/16000*0.5=0.0309688 s`; thresholds `onset=offset=0.5`, `min_duration_on=0.30 s`, `min_duration_off=0.50 s`):**
1. Per frame: argmax over 7 classes → map via decode table → binary `labels[frame][localSpk]`.
2. Per local speaker, onset/offset state-machine over frames → runs of active frames.
3. Segment time = `start_frame*scale + scale_offset` .. `end_frame*scale + scale_offset`.
4. Merge same-speaker segments with gap ≤ `min_duration_off`; drop segments < `min_duration_on`.
5. Emit `SpeakerSegment{start,end,speaker:s}`; map s → stable label/color (Speaker 1/2…).
   **Floor = single 10 s window → local slots ARE global (no stitching, no clustering).**

**Whisper greedy decode (per span, client-side):**
6. `input_ids[1,448]`=pad 50256, `[0]`=SOT 50258; `attention_mask[1,448]`=0, `[0]`=1; `idx=1`.
7. Loop while `idx<448`: run decoder → slice logits row `[(idx-1)*51865 : idx*51865]` → `next=argmax`;
   if `next==EOT(50257)` stop; else `input_ids[idx]=next; attention_mask[idx]=1; idx++`.
8. Detokenize collected ids (skip specials) via `vocab.json` → that span's text.
9. Emit `{speaker, start, end, text}`; render in timeline order.

## UI (worker's choice of visual design; functional must-haves only)
- Live/progressive scrolling transcript, each line tagged with a **speaker color + label**.
- A **who-spoke-when timeline** (speaker bands over time) — maps directly onto Kardome's positioning (nice-to-have the brief explicitly wants; cheap here since segments are already computed).
- A visible **"on-device · no cloud"** signal: an offline/airplane-mode-friendly badge AND a latency/RTF readout (per-stage timings shown on the HUD, since Dart `print` won't surface in a release device console — per CLAUDE.md §5).

## Platform targets
- iOS 16.6+, Android minSdk 24 (PyroGuard baseline).
- Known traps: (a) iOS/macOS 26.3+ CoreML-GPU MPSGraph crash — handled server-side by ZETIC; confirm served artifact isn't GPU on affected OS via device console. (b) "Benchmarked ≠ served" — budget CPU-speed until `runtimeApType=NPU` confirmed. (c) Segmentation LSTM likely lands on CPU (fine for one window). (d) Cold start: 3 model loads (2 reused Whisper + 1 new segmentation) + N decode loops — warm each with a dummy inference; pre-download.

## Validation focus (AUDIO traps — replace vision traps; Tier A test list source)
- **Sample-rate**: both families REQUIRE 16 kHz; test resampler maps N input sec → N output sec.
- **Mono downmix**: test stereo → `[1,1,160000]` / correct mel.
- **Segmentation frame count & time map**: 589 frames; test frame idx → `idx*0.016875+0.0309688 s`.
- **Powerset decode**: 7 powerset classes, NOT 7 speakers; test all 7 (esp. overlap 4/5/6) map to right local-speaker set.
- **Log-softmax**: values are log-probs; any prob threshold must `exp()` first (argmax invariant).
- **Whisper log-mel exactness**: `n_fft=400,hop=160,n_mels=80`, exactly **3000 frames**; test frame count==3000 and the clamp/scale formula vs a reference vector.
- **Greedy decode**: terminates on EOT; reads logit row at `idx-1` (not 0); buffer always 448, pad 50256, masked. Test termination + indexing.
- **Token dtype int32**: ids/mask are int32 (4-byte); test buffer byte-length.
- **Fusion attribution**: test that a span's text is tagged with the segment's speaker (the floor guarantees this by construction) — hand-built 2-segment timeline → 2 attributed lines.
- **Waveform normalization**: raw [-1,1]; test int16 → /32768 and nothing else.

## Floor → stretch ladder (what guarantees Monday, what's optional)
- **FLOOR (guaranteed video):** file-based ≤10 s clip · single segmentation window (no stitching/clustering) · diarize-then-transcribe · 2 speakers · static timeline. This is the shippable demo.
- **Stretch 1:** sliding-window segmentation for longer audio (speaker-change boundaries only; local slots not globally stable without clustering).
- **Stretch 2:** live mic instead of file.
- **Stretch 3:** FULL diarization — add a 4th Melange model (speaker-embedding: 3D-Speaker/CAM++/TitaNet, all clean static ONNX) + pure-Dart clustering for stable global identities across windows. Separate GATE-0 upload if pursued.

## Risk notes for handoff (Tier C seeds)
- **R1 (CLOSED ✅):** Whisper served shapes confirmed by fresh GATE-0 dashboard read — encoder `input_features float32[1,80,3000] → _413 float32[1,1500,384]` (READY, 85%, NPU); decoder `ids int32[1,448], enc_hidden float32[1,1500,384], enc_mask int32[1,448] → _853 float32[1,448,51865]` (READY, 81%). int32 confirmed; matches shipping-client ground truth exactly. No spec change needed.
- **R2:** `ajayshah/PyannoteSegmentation` was READY-pending at GATE 0 — confirm it reached READY before the device run.
- **R3:** segmentation LSTM may serve CPU; acceptable for one window but note `runtimeApType`.
- **R4:** running Whisper N× per clip multiplies cold-start/latency — warm models once, show progressive results in the video rather than waiting for the whole clip.
