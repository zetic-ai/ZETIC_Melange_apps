# Model selection — VoxScribe (speaker diarization / "who spoke when")

Technology family: **speaker diarization / speaker-change segmentation** (audio).
Use-case: on-device, speaker-labeled live transcript for prospect **Kardome**.
Goal: a shareable, screen-recordable **demo video before Monday KST** — optimize
for "looks impressive on camera fast", not store-readiness.

> The ASR (Whisper) half is owned by a separate Explorer. This document is ONLY
> the diarization model.

---

## TL;DR — the single biggest risk and the fallback (read this first)

**Biggest schedule risk:** full pyannote diarization is a 3-stage pipeline
(segmentation model -> speaker-embedding model -> clustering). Trying to ship the
*whole* pipeline as Melange models risks export/convert churn that eats the
weekend.

**Mitigation already executed:** the orchestrator approved a **segmentation-only
floor**, and that floor is now DE-RISKED to near-zero. The single
**pyannote/segmentation-3.0** model already exists as a pre-exported, MIT,
non-gated ONNX from k2-fsa/sherpa-onnx. I downloaded it, fixed its dynamic axes
to STATIC `[1,1,160000] -> [1,589,7]`, constant-folded out all Shape/If/Slice
logic, and confirmed the static graph is numerically identical to the original
(max abs diff = 0.0). It is sitting in this folder ready to upload.

**What the floor gives on camera:** "speaker-change segmentation" — colored
speaker bands (speaker_00 / speaker_01 / overlap) overlaid on the live transcript
with who-spoke-when timestamps. For a 2-speaker demo conversation this looks like
real diarization on screen. **This is the PRIMARY recommendation.**

**Full diarization (embedding + clustering) is a documented STRETCH only** — see
"Stretch" below. It needs a second Melange model (a speaker-embedding extractor)
plus pure-Dart clustering. The embedding model also exports cleanly and statically,
so the stretch is feasible, but it is NOT required for a shippable video and should
not block Monday.

---

## Shortlist (top 5)

| Rank | Repo / source | Downloads/Signal | License | Export path | Melange-fit notes | Score |
|------|---------------|------------------|---------|-------------|-------------------|-------|
| 1 | **k2-fsa/sherpa-onnx pyannote-segmentation-3.0** (pre-exported ONNX, GitHub release) | Very high (sherpa-onnx is the de-facto on-device speech toolkit) | **MIT**, NOT gated | Download tarball + pin dims static (DONE this session) | ~6 MB FP32; after static-fix only std ops (Conv, InstanceNorm, MaxPool, **LSTM**, MatMul, LogSoftmax); fixed 10 s window; no dynamic axes; numerically == source | **9.6** |
| 2 | pyannote/segmentation-3.0 (HF, self-export via torch.onnx) | ~millions (the reference seg model) | MIT but **HF access-GATED** | torch.onnx.export (needs torch + pyannote + accepted HF conditions) | Same architecture as #1, but ships with DYNAMIC axes by default and needs the gated weights + a torch toolchain. #1 is literally this, pre-done. | 6.0 |
| 3 | pyannote/segmentation (2.x "PyanNet", 4+1 outputs) | High | MIT, gated | torch.onnx.export | Older segmentation head; same gating + toolchain cost; superseded by 3.0 powerset model | 5.0 |
| 4 | sherpa-onnx speaker-embedding (3D-Speaker ERes2Net / wespeaker CAM++ / NeMo TitaNet) + FastClustering | High | model-dependent (3D-Speaker Apache-2.0; wespeaker Apache-2.0; NeMo CC-BY-4.0) | Pre-exported ONNX (GitHub release) | Exports clean & static; ~small. NOT a "who spoke when" engine ALONE — it's the second stage for the FULL-diarization stretch (clustering is pure-Dart). | 7.0 (stretch role) |
| 5 | NVIDIA NeMo Sortformer / `diar_sortformer_4spk` (end-to-end diarization) | Medium, newish | **CC-BY-NC** (non-commercial) on some variants | NeMo -> ONNX (immature) | True single-model end-to-end "who spoke when", but transformer with dynamic-length attention (hard to pin static), heavier, and the NC license is a GTM blocker for Kardome | 3.0 |

Honorable mention (wrong task, not scored in top 5): **snakers4/silero-vad** — a
beautifully clean static ONNX, but VAD only answers "speech vs not speech", NOT
"who spoke". It cannot do speaker change. Useful only as an optional gate to
suppress non-speech frames; do not confuse it with diarization.

---

## Winner: k2-fsa/sherpa-onnx pyannote-segmentation-3.0 (static-pinned)

Why this one over the runners-up:
- It is the realistic speaker-change engine (pyannote's PyanNet: SincNet-style
  front end + BiLSTM + linear) but **already exported to ONNX and MIT-licensed
  with NO Hugging Face gating** — so it sidesteps the single biggest cost of #2/#3
  (gated access + a torch/pyannote export toolchain), which matters for a Monday
  deadline.
- After pinning dims and constant-folding, it is a clean STATIC graph of standard
  ops with no dynamic axes and no control flow — the best possible Melange-fit.
- One fixed 10 s window per inference is cheap; even a CPU fallback is fine for the
  demo. (Full Sortformer-style end-to-end (#5) would be one model but is heavier,
  harder to make static, and partly non-commercial.)

---

## License situation (surfaced, not a footnote)

- **Winner (this artifact): MIT.** The sherpa-onnx bundle ships an MIT `LICENSE`
  (Copyright 2022 CNRS) — copied here as `LICENSE_pyannote_segmentation`. Commercial
  / demo use is permitted. Crucially, obtaining it via the **GitHub release bypasses
  the Hugging Face gating** that pyannote/segmentation-3.0 imposes on its own repo
  (the gating is an access-request form, not a restrictive license — but it would
  still block an automated/headless export).
- **Stretch embedding models:** 3D-Speaker (Apache-2.0) and wespeaker (Apache-2.0)
  are commercial-friendly; NeMo TitaNet is CC-BY-4.0 (attribution). All fine for a
  demo; prefer an Apache-2.0 one for cleanest GTM.
- **Avoid for GTM:** NeMo Sortformer NC variants (non-commercial).

---

## Export

- **Recipe:** `export.py` (this folder). It is a DOWNLOAD + static-fix recipe, not a
  torch export. Source asset:
  `https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2`
  (tarball sha256 `24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488`,
  ~6.96 MB). Used `model.onnx` (FP32), NOT `model.int8.onnx`.
- **Static-fix:** pin `N=1`, `T=160000`; `onnxsim` constant-folds out
  Shape/Slice/If/Gather/ConstantOfShape; strip phantom opset imports.
- **Input:** `x` float32[1, 1, 160000], layout (batch, channel=mono, samples).
  16 kHz mono, raw waveform ~[-1, 1]. NO app-side mean/var normalization.
- **Output:** `y` float32[1, 589, 7], layout (batch, frames, **powerset classes**).
  589 frames per 10 s window; per-frame values are **log-softmax** over 7 classes
  (exp of a row sums to ~1). Post-processing baked in? **NO** (powerset decode +
  stitching + segmentation are pure-Dart).
- **Opset:** 13 (single `ai.onnx` domain after stripping). Note EXPLORATION §4 says
  "opset ~12"; 13 is what the upstream export used and is well within Melange's
  range. Re-exporting to 12 would require the gated torch toolchain and gains
  nothing, so 13 is kept intentionally.
- **Static shapes confirmed — HOW:** loaded the saved ONNX with `onnx`, read the
  graph input/output dims (`[1,1,160000]` / `[1,589,7]` — no `dim_param`), asserted
  no `If/Shape/Slice/ConstantOfShape` nodes remain, ran `onnx.checker.check_model`
  (passed), and ran it under `onnxruntime` confirming `[1,589,7]` output that is
  bit-identical (max abs diff 0.0) to the original dynamic model. All executed this
  session.

### Embedded model metadata (carried in the ONNX, useful for the worker)
```
sample_rate            = 16000
window_size            = 160000   (10 s)
num_classes            = 7        (powerset)
num_speakers           = 3        (max local speakers per window)
powerset_max_classes   = 2        (max simultaneous speakers)
receptive_field_size   = 991      (samples ~ 0.0619375 s)  -> frame "width"
receptive_field_shift  = 270      (samples ~ 0.016875 s)   -> frame hop
model_type             = pyannote-segmentation-3.0
```
Frame rate = 16000 / 270 = **59.26 frames/s**; per-window window_shift for stitching
= 0.1 * window_size = **16000 samples (1.0 s, i.e. 90% overlap)**.
