# SPEC: RetinaDRGrade

> Product (user-facing display) name: **GradeVue** — set as the display name only
> (iOS `CFBundleDisplayName`, Android `android:label`, `MaterialApp(title:)`, app-bar /
> loading text). This is a placeholder the worker MAY keep or replace with another
> severity/grading-flavored name; it is display-name-only. Do NOT change the bundle id,
> the app folder name, or the registered Melange model name (`RetinaDRGrade`), which stay
> stable once registered. (The sibling binary screener is "FundusGate" — keep this name
> distinct from it.)

## One-line pitch
On-device diabetic-retinopathy SEVERITY grader for clinical / screening-demo prospects:
pick (or one-tap the bundled sample) a color fundus (retinal) image and get the full
5-grade DR severity — **0 No DR · 1 Mild · 2 Moderate · 3 Severe · 4 Proliferative** —
with a per-grade confidence distribution and a REFERABLE (grade ≥ 2) flag, fully offline —
the image never leaves the device. A single ViT-base forward pass; a capability / latency
proof, NOT a validated diagnostic device. This app surfaces the full severity grade; the
sibling `RetinaDRScreen` ("FundusGate") is the tiny binary referable screener.

## Model
- Source (HF repo / origin): `Kontawat/vit-diabetic-retinopathy-classification`
  (license: **apache-2.0** — clean for ZETIC GTM, no pre-ship legal gate; a plus over the
  sibling's undeclared-license MobileNetV2). Selected at GATE 0 from a 6-way validation
  bakeoff on the same 42-image held-out set (IDRiD + APTOS) — see `model_selection.md`:
  best exact-grade accuracy (0.667) and the only model with perfect referable sensitivity
  (1.00) AND high specificity (0.833) that also uses all five grades non-degenerately.
- Architecture: transformers `ViTForImageClassification` — **ViT-base**, 12-layer, 224
  input, patch 16, standard multi-head self-attention, with a 5-way classification head
  returning RAW LOGITS. Exported at opset 12 with `attn_implementation="eager"`,
  `torch.onnx.export(dynamo=False)`, static shapes, half=False; `onnx.checker` passes and
  torch-vs-onnxruntime parity was verified at export. **This is a ViT attention graph —
  exactly the class of graph that triggers Apple's GPU-compiler (MPSGraph) crash; see
  Platform targets.**
- Melange model name: **`ajayshah/RetinaDRGrade`** — the SDK requires the
  `account/project` slash form (`create(name: 'ajayshah/RetinaDRGrade')`); a bare
  `RetinaDRGrade` throws `MlangeException(3)` on-device (learned on-device this run —
  supersedes the earlier bare-name GATE-0 note). Account `ajayshah`, project
  `RetinaDRGrade`, case-sensitive. Do NOT change the bundle id, the app folder name, or
  the registered project name once registered.
- Melange version: **1**
- Input tensor: name **`pixel_values`**, `float32[1,3,224,224]`, layout **NCHW**, channel
  order **RGB**. Value range is NOT plain 0–1. Preprocessing (owned by the Dart app, NOT
  baked into the ONNX): **plain resize to 224×224** (bilinear — `Interpolation.linear`, to
  match the export/validation `ViTImageProcessor resample`) → ÷255 → [0,1] → normalize
  `(v − 0.5) / 0.5` → [−1,1] (mean = std = `[0.5, 0.5, 0.5]`, per channel). **Note this is a
  PLAIN resize-to-224, NOT the shortest-edge-256 → center-crop-224 geometry the sibling
  `RetinaDRScreen` uses — do NOT copy the sibling's crop pipeline.** See Pre-processing for
  the exact order.
- Output tensor: name **`logits`**, `float32[1,5]`, **RAW LOGITS** (one per grade), semantic
  layout: `logits[i]` = the unnormalized score for DR grade `i`. NOT softmaxed.
- Post-processing baked into ONNX? **No.** There is no softmax and no argmax in the graph —
  apply softmax then argmax in pure Dart. `predicted grade = argmax(softmax(logits))`.
- Classes / labels (index → label), **id2label is the IDENTITY map `{0:0, 1:1, 2:2, 3:3,
  4:4}`** so the argmax index IS the canonical DR grade with zero remapping:
  - `0 = No DR`
  - `1 = Mild`
  - `2 = Moderate`
  - `3 = Severe`
  - `4 = Proliferative`
  - **Referable = grade ≥ 2** (Moderate or worse). Mild (1) is NOT referable.
- modelMode to use and why: **RUN_AUTO** (default). Do NOT use `RUN_ACCURACY` (or any mode)
  as a crash workaround — no client mode steers off a crashing artifact; the server returns
  the same top-ranked candidate regardless (CLAUDE.md §5). The iOS/macOS 26.x CoreML-GPU
  (MPSGraph) crash is handled server-side by ZETIC filtering the GPU candidate. Backend /
  precision selection is server-side and not steerable from the client — only `modelMode`
  reaches the selector. Read the SERVED target + apType from the native console as the
  source of truth; this ViT attention graph is the **HIGHEST-risk** artifact in the repo for
  the GPU-compiler bug — verify on-device (see Platform targets, Validation focus).
- Melange benchmark (EXPECTED, not guaranteed-served — the served artifact on the device
  console is truth): 100% FP32 deployable; **NPU median 10.12 ms (low 4.91 ms), up to ×107
  vs CPU**; CPU median ~598 ms; **GPU median 838 ms, MAX 6.78 s (catastrophic — this is the
  GPU-compiler-bug path, avoid it)**; ~328 MB fp32 (82–328 MB across quantizations); 3
  quantizations. Treat these as the dashboard's headline, not a promise: a benchmarked NPU
  row may never be served for a given chip, and the realistic non-crashing fallback is
  CPU-speed (hundreds of ms) until the NPU/NE path is confirmed on hardware
  (`runtimeApType=NPU`). **The NPU path is the only good one here** — the GPU path is
  catastrophic (up to 6.78 s / crash) and CPU is ~60× slower than NPU.
- Validated behavior (measured ONNX vs ground truth, not eyeballed): exact-grade accuracy
  **0.667** (28/42); referable(≥2) **sensitivity 1.00** (never misses a referable eye) /
  **specificity 0.833**; per-grade recall g0 5/6, g1 5/12, g2 12/12, g3 3/6, g4 3/6
  (mid-grades Mild/Severe are the hardest, as expected for DR grading) — across a 42-image
  grade-0–4 set (IDRiD + APTOS). Cross-dataset research data, not a clinical population.

## Input source
- **Still-image UPLOAD app**, NOT live camera. A fundus image is a single framed shot, so
  there is **no live video stream, no per-frame loop, and no camera-orientation /
  rotating-buffer trap**. There is **NO camera** in this app. (The stub's "optionally
  camera capture" line is superseded — upload only.)
- Two ways to load input:
  1. Pick a fundus image from the device file picker / gallery.
  2. One-tap a **bundled sample fundus image** shipped in assets for an instant, offline
     demo (ship at least one known-referable and the validated healthy demo image so the
     discrimination is obvious on a booth device with no gallery content).
- Pixel format: decode to RGB (drop alpha). **Honor EXIF orientation** on the picked file
  before preprocessing (a still image can carry a rotation flag).
- **No boxes, no NMS, no letterbox, no anchors** — this is a whole-image classifier, not a
  detector. The plain 224×224 resize maps the entire image into the tensor; there is no
  coordinate space to invert on the output. Fundus images are effectively centered circles,
  so a plain resize (rather than a crop) is what the model was validated with.
- No microphone, no camera permission, no network. The app performs zero uploads.

## Pre-processing pipeline (ordered, exact)
1. Load the selected (or bundled sample) fundus image bytes.
2. Apply **EXIF orientation**, then decode to **RGB** (drop any alpha channel).
3. **Resize to 224 × 224 directly** (plain resize, **bilinear** — `Interpolation.linear`),
   NOT preserving aspect via a shortest-edge-256 → center-crop. This is a plain resize-to-224.
4. Convert to float32 and scale **÷255 → [0, 1]**.
5. Normalize per channel: `(v − 0.5) / 0.5` → **[−1, 1]** (mean = `[0.5, 0.5, 0.5]`,
   std = `[0.5, 0.5, 0.5]`, in R, G, B order).
6. Reorder HWC → NCHW `[1, 3, 224, 224]`, RGB channel order.
7. Flatten to a `Float32List` and wrap as
   `Tensor.float32List(data, shape: [1, 3, 224, 224])`, bound to input `pixel_values`.

(**#1 correctness trap for this classifier: use the PLAIN resize-to-224 geometry — NOT the
sibling `RetinaDRScreen`'s shortest-edge-256 → center-crop-224, and NOT a non-square squash
to some other size — and apply BOTH the ÷255 rescale AND the `(v−0.5)/0.5` normalization,
in that order. Copying the sibling's crop pipeline, using ImageNet mean/std, or dropping
either scale step silently shifts the input distribution and mis-grades.**)

## Post-processing pipeline (ordered, exact)
1. Read the output `logits` as `float32[1,5]` (5 raw logits, one per grade `0..4`).
2. Apply **softmax ONCE** over the 5 logits → a per-grade probability vector
   `probs[0..4]` (numerically stable: subtract the max logit before `exp`; `Σ probs ≈ 1`).
   The graph does not softmax — do it exactly once in Dart; do not double-apply.
3. `predicted grade = argmax(probs)` → an integer in `0..4`. **The argmax index IS the
   canonical DR grade directly (identity id2label) — do NOT remap, invert, or reorder.**
4. Derive `referable = (predicted grade ≥ 2)` (Moderate/Severe/Proliferative are referable;
   No DR / Mild are not). Exact boundary: grade 1 (Mild) → not referable; grade 2
   (Moderate) → referable.
5. Emit `Result { grade: int (0..4), perGradeProbs: List<double> (length 5),
   referable: bool }`. The full 5-way `perGradeProbs` feeds the confidence bar; the top-1
   confidence for display = `perGradeProbs[grade]`.

(No geometry, no boxes, no NMS, no letterbox, no anchors — this is a classifier producing a
single grade + a 5-way distribution. Softmax is applied ONCE in Dart; argmax uses the
identity map so the index is the grade.)

## UI
- Left to the worker for visual design. Functional must-haves:
  - The **predicted GRADE** as the primary output, shown as the number 0–4 **with its
    label** — `0 No DR` / `1 Mild` / `2 Moderate` / `3 Severe` / `4 Proliferative`
    (e.g. "Grade 3 — Severe"), large and prominent.
  - A **per-grade confidence bar** — all **5** softmax probabilities as 5 bars — so the
    full severity distribution is visible, not just the top-1.
  - A clear **REFERABLE / NOT-REFERABLE** flag (grade ≥ 2), visually distinct from the grade
    readout.
  - Show the **fundus image** that produced the grade.
  - An **inference-latency readout** (per-inference ms).
  - An **offline / on-device / "image never leaves the device — no upload"** affordance —
    this is the product's whole pitch.
  - A way to load input: file/gallery **pick** AND a one-tap **bundled sample** button.
  - **First-launch model-download / pre-warm progress.** The served model is **~328 MB
    fp32** — the first-run pull-and-cache is user-visible (a real spinner over booth Wi-Fi);
    surface progress and, ideally, a pre-download / pre-warm affordance (see Platform
    targets).
  - A **REQUIRED non-diagnostic disclaimer**, visibly on the result surface: this is a
    research / capability proof, NOT a diagnostic device; on-device inference changes
    data-residency / offline posture only; it does NOT confer or alter any FDA clearance;
    the grade is a model output on research imagery, not a clinical diagnosis.
- This app **surfaces the full 0–4 severity grade** (unlike the sibling `RetinaDRScreen`,
  which is the binary referable screener and must not show a severity grade).
- Surface any needed diagnostics (per-stage timings, the 5 raw logits, the softmax vector,
  the tensor shape) on the **UI/HUD**: in a release device build, Dart `print`/`debugPrint`
  does NOT reliably reach the native console (CLAUDE.md §5).

## Platform targets
- iOS minimum **16.6** (`IPHONEOS_DEPLOYMENT_TARGET = 16.6`, matching the repo convention);
  Android **minSdk 24**.
- Known OS traps:
  - **FP32-GPU CoreML / MPSGraph crash on iOS/macOS 26.x — HIGHEST risk in the repo for THIS
    app.** This is a **ViT-base self-attention graph**, exactly the fusion-pattern class that
    triggers Apple's GPU-compiler bug: a served FP32-GPU artifact can load cleanly
    ("BackendSelectionExecutor: success") then abort at the **first inference** inside
    MPSGraph (`MLIR pass manager failed`, SIGABRT, **uncatchable in Dart**). The dashboard
    already shows the GPU path is catastrophic here (GPU median 838 ms, MAX 6.78 s) versus
    NPU ~10 ms — so on-device this app is the **most likely** to be served a bad/GPU
    artifact. Not client-fixable: no `modelMode` steers off it (all four were tested on
    PyroGuard and returned the same crashing artifact). **The durable fix is ZETIC filtering
    the GPU candidate server-side for the affected OS.** Mandatory on-device step: **read the
    SERVED target + apType from the native console and confirm it is NOT FP32-GPU CoreML on
    iOS/macOS 26.x**; if it crashes in MPSGraph, escalate to ZETIC to filter GPU for that OS.
    Budget the **NPU path as the only good one**.
  - **First-launch DOWNLOAD of a ~328 MB fp32 model (Tier-C network / cold-start).** The
    served model must be pulled and cached on device **before the first inference**; over
    booth / conference Wi-Fi this is a long, user-visible spinner (far heavier than the
    sibling's ~17 MB). **Recommend pre-download / pre-warm the model and rehearse a
    fresh-install cold start** on the real booth network so the first demo isn't a stall.
    (Note: the raw exported ONNX is ~343 MB in the Stage-0 docs; the served/cached figure is
    ~328 MB fp32, 82–328 MB across the 3 quantizations — Melange decides serving precision
    server-side; do not pre-quantize the ONNX.)
  - **Served-artifact-is-truth vs the benchmark row:** the dashboard's fast NPU row (median
    10.12 ms) may never be served for a given chip; selection can fall back to CPU (~598 ms).
    Budget for CPU-speed as the realistic default until `runtimeApType=NPU` is confirmed on
    the device console — but treat GPU as a crash path, not a slow path.
  - **The iOS simulator is a dead end** (device-only xcframework slice, no camera anyway);
    every iteration is a signed **release** device build (debug hangs on launch on recent
    iOS/Xcode; a debug icon shows the "debug mode apps can only be launched from Flutter
    tooling" screen — expected).
- **License:** `apache-2.0` — clean, NO pre-ship legal gate (a plus over the sibling
  `RetinaDRScreen`, whose weights are `license: other` / undeclared).

## Validation focus (Tier A traps most likely for THIS model)
- **Softmax-ONCE over 5 logits:** output is RAW LOGITS `float32[1,5]` — assert softmax is
  applied downstream exactly once (`Σ probs ≈ 1`, numerically stable via max-subtraction),
  the predicted grade = the largest-logit index, and softmax is NOT double-applied.
- **argmax → grade with IDENTITY id2label:** assert the argmax index is used **AS the grade**
  (0..4) with **no remap, no inversion, no reorder**. A permutation here silently swaps
  severities.
- **`(v−0.5)/0.5` normalization exactness:** assert normalization is `(v/255 − 0.5)/0.5`
  → [−1,1], applied per channel — NOT plain ÷255, NOT ImageNet mean/std. Both the ÷255
  rescale and the mean/std normalize must be present, in that order.
- **Plain-resize-224 geometry:** assert the pipeline resizes **directly to 224×224 bilinear**
  — NOT the sibling's shortest-edge-256 → center-crop-224, and NOT a squash to any other
  size. This is the #1 silent-wrong trap and the key geometric difference from the sibling;
  a wrong resize quietly shifts probabilities.
- **Channel order:** RGB (not BGR) into channels; `(v−0.5)/0.5` applied per channel.
- **Referable-threshold derivation (grade ≥ 2):** test the boundary is exact — grade 1
  (Mild) → NOT referable; grade 2 (Moderate) → referable. Referable is derived from the
  argmax grade, not from a separate probability threshold.
- **Integration harness on the validated demo images** (`demo_images/`, measured exported-
  ONNX outputs; `argmax == grade`; reproduce `validate_demo.py` / the full-eval
  `predictions.json`). Run the pure-Dart pipeline against these and assert it reproduces the
  predicted grade and matches the full 5-way softmax within tolerance:
  - `IDRiD_g0_6389f96a.png` (GT grade 0) → predicted **0 No DR**, softmax
    `[0.982, 0.008, 0.005, 0.002, 0.003]` (top 0.982), **NOT referable**.
  - `IDRiD_g3_dd7d2789.png` (GT grade 3) → predicted **3 Severe**, softmax
    `[0.007, 0.006, 0.088, 0.810, 0.089]` (top 0.810), **referable**.
  - `IDRiD_g4_278b9ee5.png` (GT grade 4) → predicted **4 Proliferative**, softmax
    `[0.019, 0.011, 0.035, 0.125, 0.809]` (top 0.809), **referable**.
  Demo subset: 3/3 exact-grade correct; referable sens 1.00 / spec 1.00 on the subset. These
  probabilities reproduce the full-eval `predictions.json` exactly (confirming the numpy
  preprocessing matches HF's `ViTImageProcessor`), so the Dart pipeline must match them to
  prove parity.
- **Anti-degeneracy / spread sanity:** the demo grade-0 eye must return a confident No DR
  (top prob ~0.98) and the grade-3/4 eyes must peak on their true grade — a regression that
  collapses toward one mode means the pipeline (resize/normalize/softmax) is wrong.
- **Latency:** single 224×224 ViT forward pass; micro-benchmark the Dart pre/post on the hot
  path (Dart-side pre/post should be sub-millisecond; inference is dominated by the served
  artifact — EXPECTED NPU ~10 ms / CPU ~598 ms, GPU is a crash path, but the served artifact
  is truth).
- **Carry the caveats into the demo:** full 0–4 severity grade (the sibling `RetinaDRScreen`
  is the binary screener); metrics are exact-grade accuracy **0.667** / referable sensitivity
  **1.00** / specificity **0.833** on **42 research-dataset images** (IDRiD + APTOS),
  cross-dataset and not a clinical population; on-device = data-residency only, **never
  diagnostic / FDA-cleared**; **~328 MB** model + **ViT-GPU crash risk** are the two headline
  on-device caveats. License `apache-2.0` (clean — no pre-ship legal gate).
