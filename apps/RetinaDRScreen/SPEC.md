# SPEC: RetinaDRScreen

> Product (user-facing display) name: **FundusGate** — set as the display name only
> (iOS `CFBundleDisplayName`, Android `android:label`, `MaterialApp(title:)`, app-bar /
> loading text). Do NOT change the bundle id, the app folder name, or the registered
> Melange model name (`ajayshah/RetinaDRScreen`), which stay stable once registered.

## One-line pitch
On-device diabetic-retinopathy SCREENING for point-of-care camps: pick (or one-tap the
bundled sample) a color fundus (retinal) image and get a binary REFERABLE / NOT-REFERABLE
verdict with a confidence, fully offline — the image never leaves the device. Demo for
autonomous-DR-screening prospects (AEYE Health / AEYE-DS, RETINA-AI Health). A single
MobileNetV2 forward pass; a capability/latency proof, NOT a validated diagnostic device
and NOT a severity grader.

## Model
- Source (HF repo / origin): `EscvNcl/MobileNet-V2-Retinopathy` (license: `other`,
  UNDECLARED terms — FLAG, pre-ship legal gate, see Platform targets / Validation focus).
  Exported ONNX artifact: `mobilenetv2-dr-referable.onnx` (~16.6 MB served / ~17.4 MB
  raw ONNX).
- Architecture: transformers `MobileNetV2ForImageClassification` (MobileNetV2-1.4
  backbone) with a NATIVE binary head, wrapped to return raw logits. Standard mobile CNN;
  the exported opset-12 graph is all standard ops (Conv / Add / Clip / Relu6 /
  GlobalAveragePool / Gemm) — NO attention, NO dynamic axes, static shapes
  (`onnx.checker` passes; torch-vs-onnxruntime parity verified at export). Selected at
  GATE 0 from a 6-way validation bakeoff (see `model_selection.md`): smallest artifact,
  best healthy-eye specificity, native binary output.
- Melange model name: **`ajayshah/RetinaDRScreen`** — the EXACT registered string; it
  MUST include the account name and project name separated by a slash
  (`<account>/<project>`), case-sensitive. Pass this verbatim to the SDK
  `create(name: 'ajayshah/RetinaDRScreen')`. A bare `RetinaDRScreen` fails on-device with
  `MlangeException(3): Model name must include account name and project name separated by slash(/)`.
- Melange version: **1**
- Input tensor: name **`pixel_values`**, `float32[1,3,224,224]`, layout **NCHW**, channel
  order **RGB**. Value range is NOT plain 0–1. Preprocessing (owned by the Dart app, NOT
  baked into the ONNX): resize shortest edge → 256 (bilinear, preserve aspect) →
  center-crop 224×224 → ÷255 → [0,1] → normalize `(v − 0.5) / 0.5` → [−1,1]
  (mean = std = `[0.5, 0.5, 0.5]`, per channel). See Pre-processing for the exact order.
- Output tensor: name **`logits`**, `float32[1,2]`, RAW LOGITS (one per class), semantic
  layout `[0] = Nrdr logit, [1] = Rdr logit`. NOT softmaxed.
- Post-processing baked into ONNX? **No.** There is no softmax in the graph — apply
  softmax in pure Dart. `P(referable) = softmax[index 1]`.
- Classes / labels (index → label):
  - `0 = Nrdr` = NOT referable (no DR / mild, DR grade 0–1)
  - `1 = Rdr`  = REFERABLE (DR grade ≥ 2, Moderate or worse)
  - This is **BINARY**. There is NO 0–4 severity grade — do not surface one. (The 5-grade
    ViT severity option is the sibling app `RetinaDRGrade`.)
- modelMode to use and why: **RUN_AUTO** (default). No client mode steers a crashing
  artifact; GPU/MPSGraph traps are handled server-side by ZETIC (CLAUDE.md §5). Backend /
  precision selection is server-side and not steerable from the client — only `modelMode`
  reaches the selector. Read the SERVED target + apType from the native console as the
  source of truth; this plain CNN (no attention) is lower-risk for the GPU-compiler bug
  than ViT/YOLO graphs, but still verify on-device.
- Melange benchmark (EXPECTED, not guaranteed-served — the served artifact on the device
  console is truth): 100% FP32 deployable; NPU median 0.95 ms (low 0.49 ms), up to ×33 vs
  CPU; CPU median ~20 ms; ~16.6 MB; 3 quantizations. Treat these as the dashboard's
  headline, not a promise: a benchmarked NPU row may never be served for a given chip, and
  the realistic non-crashing fallback is CPU-speed (tens of ms) until the NPU/NE path is
  confirmed on hardware (`runtimeApType=NPU`).
- Validated behavior (measured ONNX vs ground truth, not eyeballed): referable
  sensitivity **0.833** (TP=20, FN=4) / specificity **0.889** (TN=16, FP=2) / binary
  accuracy **0.857** (36/42) across a 42-image grade-0–4 set (IDRiD + APTOS); 6/6
  grade-0 healthy eyes correctly called not-referable.

## Input source
- **Still-image UPLOAD app**, NOT live camera. A fundus image is a single framed shot, so
  there is no live video stream, no per-frame loop, and no camera-orientation / rotating-
  buffer trap.
- Two ways to load input:
  1. Pick a fundus image from the device file picker / gallery.
  2. One-tap a **bundled sample fundus image** shipped in assets for an instant, offline
     demo (use a known-referable or the validated healthy demo image so the discrimination
     is obvious on a booth device with no gallery content).
- Pixel format: decode to RGB (drop alpha). **Honor EXIF orientation** on the picked file
  before preprocessing (a still image can carry a rotation flag). Preserve aspect on the
  256-resize (the center-crop then takes the middle 224) — keep the retina centered; do
  NOT squash a non-square image directly to 224×224.
- No microphone, no camera permission, no network. The app performs zero uploads.

## Pre-processing pipeline (ordered, exact)
1. Load the selected (or bundled sample) fundus image bytes.
2. Apply EXIF orientation, then decode to RGB (drop any alpha channel).
3. Resize so the SHORTEST edge = 256 px, **bilinear**, preserving aspect ratio.
4. Center-crop the middle **224 × 224**.
5. Convert to float32 and scale ÷255 → [0, 1].
6. Normalize per channel: `(v − 0.5) / 0.5` → [−1, 1] (mean = `[0.5, 0.5, 0.5]`,
   std = `[0.5, 0.5, 0.5]`, in R, G, B order).
7. Reorder HWC → NCHW `[1, 3, 224, 224]`, RGB channel order.
8. Flatten to a `Float32List` and wrap as
   `Tensor.float32List(data, shape: [1, 3, 224, 224])`, bound to input `pixel_values`.

(#1 correctness trap for this classifier: the resize-256 → center-crop-224 geometry and
the `(v−0.5)/0.5` normalization must be EXACT. A direct squash-resize to 224, plain ÷255,
or ImageNet mean/std silently shifts the input distribution and mis-screens.)

## Post-processing pipeline (ordered, exact)
1. Read the output `logits` as `float32[1,2]` (2 raw logits: `[Nrdr, Rdr]`).
2. Apply **softmax** over the 2 logits → `[P(not-referable), P(referable)]` (P0 + P1 = 1).
3. `P(referable) = softmax[index 1]`. Confidence of the shown verdict = `max(P0, P1)`.
4. Decision: **REFERABLE if `P(referable) ≥ threshold`, else NOT REFERABLE.** Default
   threshold **0.5** (equivalent to argmax on 2 logits). The app MAY expose the threshold
   as a slider — a screener may prefer a lower threshold (higher sensitivity) — but the
   shipped default is 0.5; document whatever ships.
5. Emit `Result { referable: bool, pReferable: double, confidence: double }`.

(No geometry, no boxes, no NMS, no letterbox, no anchors — this is a classifier producing
a single verdict + confidence. Softmax is applied ONCE in Dart; the graph does not
softmax.)

## UI
- Left to the worker for visual design. Functional must-haves:
  - A clear **REFERABLE / NOT-REFERABLE** verdict banner as the primary output, with the
    `P(referable)` confidence (e.g. a probability bar with the 0.5 threshold marked).
  - Show the screened fundus image that produced the verdict.
  - An **inference-latency readout** (per-inference ms).
  - An **offline / on-device / "image never leaves the device — no upload"** affordance —
    this is the product's whole pitch.
  - A way to load input: file/gallery **pick** AND a one-tap **bundled sample** button.
  - A **REQUIRED non-diagnostic disclaimer**, visibly on the result surface: this is a
    research/capability proof, NOT a diagnostic device; on-device inference changes
    data-residency / offline posture only; it does NOT confer or alter any FDA clearance;
    it is a binary screen, not a severity grade, and not a clinical diagnosis.
- Do NOT present a 0–4 severity grade — the model does not output one.
- Surface any needed diagnostics (per-stage timings, the two raw logits, tensor shape) on
  the UI/HUD: in a release device build, Dart `print`/`debugPrint` does NOT reliably reach
  the native console (CLAUDE.md §5).

## Platform targets
- iOS minimum **16.6** (`IPHONEOS_DEPLOYMENT_TARGET = 16.6`, matching the repo's
  FireDetectionYOLO convention); Android **minSdk 24**.
- Known OS traps:
  - **FP32-GPU CoreML / MPSGraph crash on iOS/macOS 26.x**: a served FP32-GPU artifact can
    load cleanly then abort at the first inference inside Apple's GPU compiler (SIGABRT,
    uncatchable in Dart). Not client-fixable — no `modelMode` steers off it; the durable
    fix is ZETIC filtering the GPU candidate server-side for the affected OS. This is a
    plain CNN with no attention, so it is **lower-risk** for the fusion bug than ViT/YOLO
    graphs, but still **read the SERVED target + apType from the native console** and
    confirm it is not GPU on affected OS versions.
  - **Served-artifact-is-truth vs the benchmark row**: the dashboard's fast NPU row
    (median 0.95 ms) may never be served for a given chip; selection can fall back to
    `TFLITE_FP16 / CPU` (tens of ms). Budget for CPU-speed as the realistic default until
    `runtimeApType=NPU` is confirmed on the device console.
  - The iOS simulator is a dead end (device-only xcframework slice); every iteration is a
    signed **release** device build (debug hangs on launch on recent iOS/Xcode).
- **LICENSE pre-ship legal gate (blocking GTM):** `EscvNcl/MobileNet-V2-Retinopathy`
  declares `license: other` with NO stated terms; the base `google/mobilenet_v2` is
  Apache-2.0 but the fine-tuned DR weights' redistribution / commercial terms are
  UNDECLARED. Clear the license (author / training-data terms) before shipping. If it
  cannot be cleared, the drop-in alternates from the same bakeoff are the Apache-2.0
  transformers (#2 Kontawat ViT, #3 Augusto SwinV2, #6 rafalosa ViT) — each far larger and
  a worse Melange fit, so clearing the winner's license is the preferred path.

## Validation focus (Tier A traps most likely for THIS model)
- **Softmax correctness (2-logit → probability):** output is RAW LOGITS `[1,2]` — assert
  softmax is applied downstream (`P0 + P1 ≈ 1`), that the verdict = argmax matches the
  larger logit, and that softmax is NOT double-applied.
- **Resize-256 → center-crop-224 geometry exactness:** assert the pipeline is
  shortest-edge → 256 (bilinear) THEN center-crop 224 — NOT a direct squash-resize to 224.
  This is the #1 silent-wrong trap; a wrong crop quietly shifts the input distribution.
- **`(v−0.5)/0.5` normalization exactness:** assert normalization is `(v/255 − 0.5)/0.5`
  → [−1,1], NOT plain ÷255 and NOT ImageNet mean/std. A wrong normalization silently
  degrades everything.
- **Threshold boundary at 0.5:** test the decision flips exactly at
  `P(referable) = threshold` — Nrdr just below, Rdr just at/above.
- **Anti-degeneracy (healthy eye not over-flagged):** a known clearly-healthy (grade-0)
  fundus must come back **NOT REFERABLE** with low `P(referable)` — the model's key
  validated strength (6/6 on grade-0). A regression here means the pipeline is wrong.
- **Integration harness on the validated demo images** (`demo_images/`, measured ONNX
  outputs):
  - `IDRiD_g0_630e24b6.png` (grade 0) → NOT REFERABLE, logits `[10.11, −0.66]`,
    `P(referable) ≈ 0.0000`.
  - `IDRiD_g3_ca10d891.png` (grade 3) → REFERABLE, logits `[−2.72, 2.75]`,
    `P(referable) ≈ 0.9958`.
  - `IDRiD_g4_ce3e6abe.png` (grade 4) → REFERABLE, logits `[−2.30, 2.29]`,
    `P(referable) ≈ 0.9900`.
  Run the Dart pipeline against these and assert it reproduces the decisions and matches
  `P(referable)` within tolerance. Aggregate reference: sensitivity 0.833 / specificity
  0.889 / accuracy 0.857.
- **Channel order:** RGB (not BGR) into channels; the `(v−0.5)/0.5` normalization applied
  per channel.
- **Label mapping:** index `0 = Nrdr` (not-referable), index `1 = Rdr` (referable). Do not
  invert.
- **Latency:** single 224×224 CNN forward pass; micro-benchmark the Dart pre/post on the
  hot path (should be sub-millisecond Dart-side; inference dominated by the served
  artifact — EXPECTED NPU ~0.95 ms / CPU ~20 ms, but the served artifact is truth).
- **Carry the caveats into the demo:** binary only (no severity — that is the sibling
  `RetinaDRGrade`); metrics are on 42 research-dataset images (IDRiD/APTOS), not a clinical
  population; on-device = data-residency only, never diagnostic / FDA-cleared; license
  `other` (undeclared) → pre-ship legal gate.
