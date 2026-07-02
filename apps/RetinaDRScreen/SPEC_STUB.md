# SPEC: RetinaDRScreen

## One-line pitch
On-device diabetic-retinopathy SCREENING for point-of-care camps: capture a color
fundus (retinal) image and get a binary REFERABLE / NOT-REFERABLE verdict with a
confidence, fully offline — the image never leaves the device. Demo for
autonomous-DR-screening prospects (AEYE Health / AEYE-DS, RETINA-AI Health). Single
MobileNetV2 forward pass; NOT a validated diagnostic device, NOT a severity grader.

## Model
- Source (HF repo / origin): EscvNcl/MobileNet-V2-Retinopathy (license: `other` — FLAG,
  see below). Exported ONNX: `mobilenetv2-dr-referable.onnx`.
- Architecture: transformers MobileNetV2ForImageClassification (MobileNetV2-1.4
  backbone) with a NATIVE binary head. ~17.4 MB ONNX; standard mobile CNN, exported
  graph is all standard ops (Conv/Add/Clip/Relu6/GlobalAveragePool/Gemm) — no attention,
  no dynamic axes. Selected at GATE 0 from a 6-way validation bakeoff (see
  model_selection.md): smallest artifact, best healthy-eye specificity, native binary
  output. Validated: referable sensitivity 0.833 / specificity 0.889 / binary accuracy
  0.857; 6/6 healthy (grade-0) eyes correctly called not-referable.
- Melange model name: `<<FILL AT GATE 0>>` (requested: ajayshah/RetinaDRScreen)
- Melange version: `<<FILL AT GATE 0>>` (requested: 1)
- Input tensor: float32[1,3,224,224], NCHW, RGB. Value range NOT plain 0-1 — the
  MobileNetV2 pipeline: resize shortest-edge->256 (bilinear), center-crop 224,
  *1/255 -> [0,1], normalize (v-0.5)/0.5 -> [-1,1] (mean=std=[0.5,0.5,0.5]).
  - Served shape: `<<FILL AT GATE 0>>`
- Output tensor: float32[1,2], RAW LOGITS (2 values, one per class). NOT softmaxed.
  - Served shape: `<<FILL AT GATE 0>>`
- Post-processing baked into ONNX? No. No softmax in the graph — apply softmax in pure
  Dart. P(referable) = softmax[index 1].
- Classes / labels (index -> label):
  0 = Nrdr (NOT referable — no DR / mild, DR grade 0-1)
  1 = Rdr  (REFERABLE — DR grade >= 2, Moderate or worse)
  This is BINARY. There is NO 0-4 severity grade — do not surface one.
- modelMode to use and why: RUN_AUTO (default). No client mode steers a crashing
  artifact; GPU/MPSGraph traps are handled server-side by ZETIC. See CLAUDE.md S5.
  (Lower risk here than ViT/YOLO graphs — this is a plain CNN with no attention — but
  still read the SERVED target+apType from the native console.)

## Input source
- Primary: still image from a fundus/retinal camera or the device gallery (a fundus
  image is a single framed shot, not a live video stream) — a "capture / pick image"
  flow, then screen on tap. A live-camera mode is optional; this is per-image
  classification, not a per-frame detector.
- Pixel format: decode to RGB (drop alpha). Preserve aspect on the 256-resize (the
  center-crop then takes the middle 224) — keep the retina centered, do NOT squash a
  non-square image to 224x224 directly.

## Pre-processing pipeline (ordered, exact)
1. Load the fundus image bytes; decode to RGB.
2. Resize so the SHORTEST edge = 256 px (bilinear), preserving aspect ratio.
3. Center-crop the middle 224 x 224.
4. Convert to float32, scale * 1/255 -> [0,1].
5. Normalize per channel: (v - 0.5) / 0.5 -> [-1,1] (mean=[0.5,0.5,0.5],
   std=[0.5,0.5,0.5], R,G,B order).
6. Reorder HWC -> NCHW [1,3,224,224], RGB channel order.
7. Flatten to Float32List, wrap as Tensor.float32List(data, shape:[1,3,224,224]).

## Post-processing pipeline (ordered, exact)
1. Read `logits` as float32[1,2] (2 raw logits).
2. Softmax over the 2 logits -> [P(not-referable), P(referable)].
3. P(referable) = softmax[index 1]; confidence of the shown verdict = max(P0, P1).
4. Decision: REFERABLE if P(referable) >= threshold, else NOT REFERABLE.
   Default threshold 0.5 (argmax). The app MAY expose the threshold as a slider — a
   screener may want a lower threshold (higher sensitivity) in practice; document
   whatever default ships.
5. Emit Result{referable(bool), pReferable, confidence}.

## UI
- Left to the worker. Functional must-haves:
  - Show the screened fundus image.
  - A clear REFERABLE / NOT-REFERABLE verdict banner (the primary output), with the
    P(referable) confidence (e.g. a probability bar).
  - Offline / on-device / "image never leaves the device — no upload" affordance
    (this is the product's whole pitch).
  - An inference-latency readout.
  - A visible "research/demo, not a diagnosis" disclaimer (no clinical claim; binary
    screen only, not a severity grade).
- Do NOT present a 0-4 severity grade — the model does not output one.

## Platform targets
- iOS 16.6+, Android minSdk 24.
- Known OS traps: FP32-GPU CoreML artifact can crash in MPSGraph on iOS/macOS 26.3+;
  not client-fixable (no modelMode avoids it) — read the SERVED target+apType from the
  native console and confirm it is not GPU on affected OS versions. This is a plain CNN
  (no attention), so it is lower-risk for that fusion bug than YOLO/ViT graphs, but
  still verify. Realistic non-crashing fallback is TFLITE_FP16/CPU (tens-to-hundreds of
  ms), not guaranteed NPU.

## Validation focus (Tier A traps most likely for THIS model)
- Softmax correctness: output is RAW LOGITS[1,2] — assert softmax is applied downstream
  (P0+P1 == 1) and that the verdict = argmax matches the larger logit; confirm softmax
  is NOT double-applied.
- The 256->224 preprocessing: assert resize is shortest-edge->256 THEN center-crop 224
  (NOT a direct squash-resize to 224), and that normalization is (v/255 - 0.5)/0.5,
  NOT plain /255 and NOT ImageNet mean/std. This is the #1 silent-wrong trap — the wrong
  crop or normalization quietly shifts the input distribution and mis-screens.
- Threshold boundary: test the decision flips exactly at P(referable) = threshold
  (default 0.5); Nrdr just below, Rdr just at/above.
- Healthy-eye not over-flagged: feed a known clearly-healthy (grade-0) fundus and assert
  it comes back NOT-REFERABLE with low P(referable) — the model's key validated strength
  (6/6 on grade-0); a regression here means the pipeline is wrong.
- Channel order: RGB (not BGR) into channels; the (v-0.5)/0.5 normalization applied
  per-channel.
- Label mapping: index 0 = Nrdr (not-referable), index 1 = Rdr (referable). Do not
  invert.
- Latency: single 224x224 CNN forward pass; benchmark the Dart pre/post on the hot path
  (should be sub-millisecond Dart-side; inference dominated by the served artifact).
