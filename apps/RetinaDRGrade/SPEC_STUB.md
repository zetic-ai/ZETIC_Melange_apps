# SPEC: RetinaDRGrade  (STUB — GATE-0 fields left blank for the human)

> Stage-0 stub. Everything the ONNX already reveals is filled in. The GATE-0 fields
> (Melange served name/version + served shapes + modelMode) are BLANK until the human
> uploads per `melange_upload.md` and pastes the dashboard echo back. The worker does not
> start app code until those are filled.

## One-line pitch
On-device diabetic-retinopathy SEVERITY grader: point it at a fundus image and it returns
the 5-grade DR severity (0 No DR .. 4 Proliferative) with per-grade confidence and a
referable flag — offline, no upload, for clinical/screening-demo prospects.

## Model
- Source (HF repo / origin): Kontawat/vit-diabetic-retinopathy-classification
- Architecture: ViT-base (ViTForImageClassification), 12-layer, 224 input, patch 16.
- Melange model name: ajayshah/RetinaDRGrade   <!-- confirm served name at GATE 0 -->
- Melange version: 1                            <!-- confirm at GATE 0 -->
- Input tensor: float32[1,3,224,224], NCHW, RGB, values in [-1,1] after normalization
  (resize 224 bilinear -> /255 -> normalize mean/std [0.5,0.5,0.5]).
- Output tensor: float32[1,5] = RAW LOGITS over grades. dim1 index i == canonical grade i.
- Post-processing baked into ONNX? NO. Softmax + argmax done downstream in Dart.
- Classes / labels: [0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative].
  id2label is the IDENTITY map, so argmax index == grade with no remap. Referable = grade >= 2.
- modelMode to use and why: RUN_AUTO (default). Do NOT use RUN_ACCURACY as a crash
  workaround. <!-- confirm served modelMode / served shapes at GATE 0 -->

## Input source
- File / gallery pick of a fundus photograph (demo), optionally camera capture of a fundus.
  This is a still-image grader, not a live video stream.
- Pixel format: decode to RGB, 8-bit.
- Orientation handling: fundus images are orientation-agnostic circles; center-crop/resize
  is fine. No rotation logic needed (unlike the camera-detector apps).

## Pre-processing pipeline (ordered, exact)
1. Decode image to RGB.
2. Resize to 224x224 with BILINEAR interpolation.
3. Convert to float32 and rescale /255.0 -> [0,1].
4. Normalize per channel: (x - 0.5) / 0.5  (mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]) -> [-1,1].
5. Reorder HWC -> CHW, add batch dim -> [1,3,224,224].
6. Flatten to Float32List, wrap as Tensor.float32List(shape:[1,3,224,224]).

## Post-processing pipeline (ordered, exact)
1. Read output logits Float32List of length 5.
2. Softmax over the 5 logits -> per-grade probability (numerically stable: subtract max).
3. argmax -> predicted grade in 0..4 (index == grade directly; NO id2label remap).
4. referable = (predicted grade >= 2). (Optionally also flag referable if
   P(grade>=2) = p2+p3+p4 exceeds a chosen threshold — decide with the human.)
5. Surface the full 5-way probability vector for the confidence bar.

## UI (functional must-haves; visual design left to worker)
- Predicted DR grade, large and named (e.g. "Grade 3 — Severe").
- Per-grade confidence bar (all 5 softmax probabilities), so the distribution is visible,
  not just the top-1.
- A clear REFERABLE / not-referable flag (grade >= 2), visually distinct.
- Fully offline / no-upload messaging (data stays on device).
- Inference latency readout.
- First-launch model-download progress (the model is ~343 MB — the download is user-visible).

## Platform targets
- iOS minimum / Android minSdk: per repo standard (confirm with worker).
- Known OS traps for this artifact: standard ViT attention heads are exactly the shape that
  triggered the iOS/macOS 26.3+ CoreML-GPU (MPSGraph) crash on PyroGuard — expect the
  server-side GPU filter to apply; plan for a CPU-speed fallback until an NE artifact is
  confirmed on the device console. ~343 MB download/footprint is a real UX consideration.

## Validation focus (the correctness traps most likely for THIS model)
- Softmax + argmax correctness: pure-Dart softmax must match the reference (parity vs
  `validate_demo.py` on the demo images — probs like 0.9823 / 0.8104 / 0.8091 must reproduce).
- Identity id2label mapping: argmax index must be used AS the grade — no accidental remap.
- Referable threshold: grade >= 2 boundary must be exact (Mild=1 is NOT referable;
  Moderate=2 IS).
- Normalization exactness: the [0.5]/[0.5] mean-std (mapping to [-1,1]) AND the /255 rescale
  must both be applied, in that order — dropping either silently corrupts predictions.
- Resize interpolation: 224 BILINEAR (matches ViTImageProcessor resample=2); a different
  filter shifts probabilities.
