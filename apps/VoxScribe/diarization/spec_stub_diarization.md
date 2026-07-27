# SPEC (stub): VoxScribe — diarization half

> Pre-drafted by the Stage-0 Explorer. Everything the ONNX reveals is filled in.
> GATE-0 fields (Melange name/version, served shapes) are left BLANK for the human
> to confirm after upload. This is the diarization model ONLY; ASR (Whisper) is a
> separate Explorer/spec and is merged by the orchestrator.

## One-line pitch
On-device speaker-change segmentation that paints "who spoke when" speaker bands
onto a live transcript — the diarization layer of VoxScribe, for prospect Kardome.

## Model
- Source (origin): k2-fsa/sherpa-onnx pre-exported **pyannote/segmentation-3.0**
  (GitHub release `speaker-segmentation-models`), then static-pinned. See
  `model_selection.md` / `export.py`.
- Architecture: PyanNet — SincNet-style conv front end + 2x BiLSTM + linear, with a
  **powerset** classification head (max 3 local speakers, max 2 simultaneous).
- Melange model name: __________________  (expected `ajayshah/PyannoteSegmentation`) [GATE 0 — confirm]
- Melange version:    __________________  (expected `1`)                              [GATE 0 — confirm]
- Input tensor: `x` float32[1, 1, 160000]; layout (batch, channel=mono, samples);
  16 kHz mono raw waveform, value range ~[-1, 1]; **NO** mean/variance normalization.
- Output tensor: `y` float32[1, 589, 7]; layout (batch, frames, **powerset classes**).
  Per-frame values are **log-softmax** over 7 classes (so `exp(row)` sums to ~1).
  589 frames per 10 s window. Frame hop 270 samples (16.875 ms) -> 59.26 frames/s.
- Served input/output shapes (dashboard echo): __________________ [GATE 0 — confirm == [1,1,160000] -> [1,589,7]]
- Post-processing baked into ONNX? **No.** Powerset decode + window stitching +
  segment extraction are pure-Dart (below).
- Classes / labels: 7 powerset classes. Decode table (derived from the model's
  `num_speakers=3, powerset_max_classes=2`):
  | class idx | active local speakers |
  |-----------|-----------------------|
  | 0 | {} (silence / no speaker) |
  | 1 | {spk0} |
  | 2 | {spk1} |
  | 3 | {spk2} |
  | 4 | {spk0, spk1} |
  | 5 | {spk0, spk2} |
  | 6 | {spk1, spk2} |
  "spk0/1/2" are **per-window LOCAL** slots, not global identities (see traps).
- modelMode to use and why: **RUN_AUTO** (per CLAUDE.md §5; no client mode steers off
  the iOS-26 GPU crash, which ZETIC handles server-side). Confirm served
  `runtimeApType` on the device console — LSTM may land on CPU, which is fine here.

## Input source
- Mic (live) or an audio file for the demo clip.
- Requested format: **16 kHz, mono, float32 PCM in [-1, 1]**. If the capture device
  delivers another rate (e.g. 44.1/48 kHz) or stereo, resample to 16 kHz and downmix
  to mono BEFORE windowing.
- Orientation handling: N/A (audio).

## Pre-processing pipeline (ordered, exact)
1. Acquire PCM; if stereo, downmix to mono (take ch0 or average — the reference uses
   ch0).
2. If sample rate != 16000, resample to 16000 (linear/sinc; quality matters less than
   correctness of the rate).
3. Ensure float32 in [-1, 1] (if source is int16, divide by 32768). NO further
   normalization.
4. Window into fixed 10 s frames of **160000 samples**.
   - **Floor (recommended for the demo): single window.** Keep the demo clip <= 10 s
     (or take one 10 s window). Pad the tail with zeros to 160000 if short. One window
     => the 3 local-speaker slots are STABLE for the whole clip (no stitching, no
     clustering) => clean speaker_00 / speaker_01 bands on camera.
   - **Sliding window (longer audio):** hop = `0.1 * 160000` = **16000 samples (1.0 s,
     90% overlap)**. Last partial chunk is zero-padded to 160000. (This is what enables
     a longer transcript but introduces the cross-window permutation trap below.)
5. Shape each window to [1, 1, 160000] Float32List; wrap as Tensor.float32List.

## Post-processing pipeline (ordered, exact)
Constants from the model metadata:
- `frame_shift = 270 samples` => `scale = 270/16000 = 0.016875 s/frame`
- `frame_size  = 991 samples` => `scale_offset = 991/16000 * 0.5 = 0.0309688 s`
- thresholds (pyannote defaults): `onset=0.5, offset=0.5, min_duration_on=0.30 s,
  min_duration_off=0.50 s`

### A) Powerset decode (per window, every frame)
1. For each of 589 frames, `argmax` over the 7 classes (log-softmax, so argmax of the
   logits == argmax of the probs).
2. Map the winning class to its local-speaker set via the decode table above ->
   binary activity `labels[frame][localSpk] in {0,1}` for localSpk in 0..2.

### B-floor) Single-window segments (the demo floor — no stitching, no clustering)
3. For each local speaker s in 0..2, walk frames; a run of active frames is a segment.
   Use onset to enter / offset to leave the active state.
4. Segment time = `start_frame * scale + scale_offset` .. `end_frame * scale +
   scale_offset`.
5. Merge segments of the SAME speaker separated by a gap <= `min_duration_off`.
6. Drop segments shorter than `min_duration_on`.
7. Emit `SpeakerSegment{ start, end, speaker: s }`. Map s -> a stable color/label
   (Speaker 1 / Speaker 2 ...). Align against the transcript by timestamp.

### B-stretch) Sliding-window stitching (longer audio, segmentation-only)
- Aggregate overlapping windows onto a global frame timeline of
  `num_frames = (window_size + (num_chunks-1)*window_shift)/frame_shift + 1`, summing
  per-(global-frame, local-speaker) activity and the per-frame speaker COUNT
  (the reference `speaker_count` averages activity across overlapping windows).
- CAVEAT: local-speaker slots are NOT consistent across windows. Without clustering,
  use this only for "how many speakers are active now" / speaker-change boundaries,
  NOT stable global identities. Stable global identity needs the stretch below.

### C-stretch) FULL diarization (segmentation + embedding + clustering)
Documented, NOT required for the video. Adds a SECOND Melange model (a speaker-
embedding extractor) + pure-Dart clustering:
1. For each window+local-speaker with enough active frames, gather its audio span and
   run the embedding model -> a fixed-dim speaker vector.
2. Cluster all embeddings (cosine) with a fixed `num_clusters` (e.g. 2 for a known
   2-person demo) or a distance threshold -> GLOBAL speaker ids.
3. Relabel local slots to global cluster ids, re-stitch overlapping windows, then run
   the onset/offset + min-duration segmentation (B-floor steps 3-7) on the global,
   re-labeled timeline.
- Embedding model options (sherpa-onnx `speaker-recongition-models` release — note the
  upstream tag is misspelled "recongition"; all return 302/exist):
  - `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` (Apache-2.0)
  - `wespeaker_en_voxceleb_CAM++.onnx` (Apache-2.0)
  - `nemo_en_titanet_small.onnx` (CC-BY-4.0)
  These export clean & static; clustering is pure-Dart. This is a separate GATE-0
  upload if the stretch is pursued.

## UI
- Worker's choice. Functional must-haves: speaker-labeled bands/colors on the live
  transcript (Speaker 1 / Speaker 2 / overlap), who-spoke-when timestamps, and a
  small inference-latency readout. For the floor, a 2-speaker scripted exchange in a
  single <=10 s window is the most reliable thing to screen-record.

## Platform targets
- iOS minimum 16.6+, Android minSdk 24 (match PyroGuard baseline).
- Known traps for THIS artifact: LSTM may force a CPU fallback (acceptable — one
  10 s window is cheap). Always read the SERVED artifact (target + apType) from the
  native device console, not the dashboard headline; treat CPU-speed as the realistic
  default until an NPU artifact is confirmed.

## Validation focus (AUDIO correctness traps — these REPLACE the vision traps)
- **Sample-rate mismatch:** model REQUIRES 16 kHz. Feeding 44.1/48 kHz without
  resampling silently shifts every timestamp and wrecks segmentation. Unit-test the
  resampler maps N input seconds -> N output seconds.
- **Mono vs stereo:** input channel dim is 1. Downmix stereo to mono (ch0 or average)
  before windowing; test a stereo source produces [1,1,160000].
- **Window length & HOP/overlap:** window = 160000 samples (10 s); sliding hop =
  16000 (1 s, 90% overlap); tail zero-padded to 160000. Test the stitching math
  (`num_frames` formula) and that the single-window floor needs NO stitching.
- **Output semantics / frame rate:** 589 frames/window, frame hop 16.875 ms, frame
  rate 59.26 fps; segment time = `frame*0.016875 + 0.0309688`. Test a known frame
  index maps to the expected seconds and aligns with the transcript clock.
- **Powerset vs per-speaker decode:** output is 7 POWERSET classes, NOT 7 speakers and
  NOT per-speaker sigmoids. Must argmax-then-map via the decode table; test all 7
  classes map to the right speaker set (esp. overlap classes 4/5/6).
- **Log-softmax, not probabilities:** values are log-probs; `exp(row).sum() ~= 1`.
  argmax is invariant, but any thresholding on "probability" must exponentiate first.
- **Waveform normalization:** raw [-1, 1] float; do NOT z-score/normalize. Test int16
  sources are divided by 32768 and nothing else.
- **Number-of-speakers handling:** the model outputs at most 3 LOCAL speakers per
  window (max 2 simultaneous). Local slots are per-window and NOT globally consistent
  across sliding windows — global identity needs the embedding+clustering stretch.
  For the floor, restrict the demo to a single <=10 s window so local == global.

## GATE-0 fields to confirm (leave blank until the human pastes back)
- Registered Melange model name + version: __________________
- Served input shape:  __________________ (expect float32[1,1,160000])
- Served output shape: __________________ (expect float32[1,589,7])
- modelMode: __________________ (default RUN_AUTO)
- Served runtimeApType (NPU/GPU/CPU): __________________
