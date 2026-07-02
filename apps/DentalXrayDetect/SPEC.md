# SPEC: DentalXrayDetect

## One-line pitch
On-device dental-radiograph analyzer that outlines caries, periapical lesions, and impacted
teeth on a still X-ray image directly in the operatory (workstation, tablet, or phone) — the
radiograph and any PHI never leave the practice. It is a capability-proof wedge against
cloud-SaaS chairside dental AI (Overjet / VideaHealth): on-device inference changes DATA
RESIDENCY only. It is explicitly **not** a diagnostic device and confers **no** FDA status.

## Model
- Source (HF repo / origin): `liodon-ai/dental-panoramic-detector`, weights `best.pt`;
  re-exported to ONNX via `export.py` (Ultralytics `YOLO(best.pt).export(format='onnx',
  imgsz=640, opset=12, simplify=True, dynamic=False, half=False)`) to guarantee opset 12 +
  static axes. Trained on 9,928 dental **panoramic** radiographs (DENTEX + OralXrays-9).
- Architecture: YOLO11n (Ultralytics), anchor-free single detect head. 2.58 M params,
  6.3 GFLOPs, ~10 MB ONNX. Standard, static-shape, standard-op CNN export — the cleanest
  Melange fit of the shortlist (same family as PyroGuard / AerialDetect).
- Melange model name: `ajayshah/DentalXRayDetect` — the SDK requires the fully-qualified
  `account/project` form (account `ajayshah`, project `DentalXRayDetect`). A bare project name
  fails on-device with `MlangeException(3): Model name must include account name and project name
  separated by slash(/)`. Pass this to the SDK `create(name: ...)` **exactly**, matching case.
  Note the **capital `R`** in "XRay": the project part DIFFERS from the app **folder** name
  `DentalXrayDetect` (lowercase `r`) — do **not** substitute the folder name.
- Melange version: `1`
- Input tensor: `images`, `float32[1,3,640,640]`, layout **NCHW**, channel order **RGB**,
  value range **0.0–1.0** (pixels ÷ 255), letterboxed to 640×640.
- Output tensor: `output0`, `float32[1,7,8400]`, **channel-major** — dimension 1 (size 7) is
  the channel axis, dimension 2 (size 8400) is the anchor axis. Per anchor the 7 channels are
  `[cx, cy, w, h, s0, s1, s2]` = 4 box coordinates + 3 class scores. Box coordinates are in
  **PIXEL space of the 640×640 letterbox frame** (not normalized 0–1; verified box rows range
  ~5.8–636). 8400 anchors = 80²+40²+20² across the /8, /16, /32 strides. YOLO11 has **no**
  separate objectness channel — 7 = 4 box + 3 class, nothing more.
- Post-processing baked into ONNX?
  - **NMS is NOT baked in** — implement per-class NMS in pure Dart.
  - **Sigmoid IS baked in** — the 3 class scores are already sigmoid-activated (verified
    in-graph via onnxruntime: class rows lie within [0,1]). **Do NOT re-apply sigmoid in Dart**
    — a second activation would crush every score toward 0.5–0.62 and silently break the
    threshold. This is the single highest-risk trap for this model.
- Classes / labels (3, verified from checkpoint `model.names`):
  `0 caries`, `1 periapical_lesion`, `2 impacted_tooth`.
- modelMode to use and why: **RUN_AUTO** (default). No client `modelMode` steers off a crashing
  artifact — that was proven on PyroGuard, where all four modes returned the same FP32-GPU
  candidate. The iOS/macOS-26 GPU/MPSGraph trap is handled **server-side** by ZETIC filtering
  GPU for the affected OS. RUN_AUTO will pick a Neural-Engine artifact automatically if/when
  ZETIC serves one. See CLAUDE.md §5.
- Benchmark (record as EXPECTED, **not** guaranteed-served): 100% FP32 deployable; NPU median
  3.71 ms (low 1.35 ms), up to ×72 vs CPU; CPU median ~123 ms; smallest served memory ~11 MB;
  3 quantizations available. **The served artifact (`target` + `apType`, read from the native
  console) is the source of truth — not this benchmark row.** Budget for CPU-speed as the
  realistic default until the NPU path (`runtimeApType=NPU`) is confirmed on hardware.

## Input source
- **File / gallery image — NOT a live camera.** A dental X-ray is a static radiograph; there
  is no video pipeline. The app runs **single-shot inference on one selected still image**:
  either a file/gallery pick, a photo of a mounted X-ray, or a bundled sample radiograph
  shipped in assets for a zero-setup trade-show demo. Because there is no camera, there is
  **no camera-orientation / rotating-buffer trap** (the class of bug that dominated PyroGuard).
- Pixel format: decode the picked image to RGB. Radiographs are effectively grayscale, but the
  model expects 3 channels — replicate luma across R, G, B (or decode RGB directly).
- Orientation handling: honor **EXIF orientation** on the decoded file before letterboxing;
  that is the only orientation concern. There is no live buffer to measure. The one geometry
  task that remains is mapping the letterbox transform **and its inverse** correctly so boxes
  land on the displayed image (see post-processing) — the display may be fit/letterboxed inside
  the widget, so the letterbox inverse must target the *original image* space and then be
  composed with the image→widget display transform.
- Device targets: dental-operatory workstation, tablet, and phone. The pipeline is
  resolution-agnostic (source radiographs are large, e.g. ~2872×1504); everything is normalized
  through the 640 letterbox, so device form factor affects only display scaling, not inference.

## Pre-processing pipeline (ordered, exact)
1. Load the radiograph image bytes; decode; apply **EXIF orientation**.
2. Convert to RGB: if the source is single-channel grayscale, **replicate luma to all 3
   channels**; if already RGB, keep as-is.
3. **Letterbox-resize to 640×640**: scale by `r = min(640/W, 640/H)` (preserve aspect ratio),
   using **bilinear** interpolation to match the validated harness / Ultralytics letterbox
   (`cv2.INTER_LINEAR`) so on-device confidences track the measured demo recall (nearest-neighbour
   would shift borderline detections at the 0.45 gate). Pad the remainder with gray (value **114**
   in 0–255 → **0.447** after ÷255); the resized image is centered with integer `floor((640−nw)/2)`
   offsets, mirroring the harness. **Record the exact `r` (scale) and `(padX, padY)`** — they are
   required verbatim for the letterbox inverse in post-processing.
4. Normalize: divide every channel value by **255.0** → 0.0–1.0.
5. Reorder to **NCHW** `[1,3,640,640]` (channel-planar: all R, then all G, then all B).
6. Flatten to a `Float32List` and wrap as `Tensor.float32List(data, shape: [1,3,640,640])`.

## Post-processing pipeline (ordered, exact)
1. Read the output as **channel-major `[1,7,8400]`**: to get anchor `a`, index channel `c` at
   flat offset `c * 8400 + a` — i.e. **stride across the 8400 anchor axis, NOT across the 7**.
   (Reading it row-major — 7 values contiguous per anchor — is the classic decode bug and
   produces garbage boxes.)
2. For each of the 8400 anchors, the 4 box coords are channels 0..3 `[cx, cy, w, h]` and the
   3 class scores are channels 4..6 `[s0, s1, s2]`. Take **`max` of the 3 class scores** and
   its argmax index as the anchor's label.
3. **Do NOT apply sigmoid.** The class scores are already sigmoid-activated in-graph (0–1). Use
   them as-is. (Re-activating is the double-activation bug — see Model.)
4. Threshold **before** any box geometry: keep an anchor only where its max class score
   **≥ 0.45**. `0.45` is the **validated operating point** (model card + measured on the demo
   set). At conf 0.25 caries over-fires badly on adjacent healthy teeth (precision collapses to
   ~0.33); do not lower the default.
5. Convert surviving boxes `cxcywh → x1y1x2y2` (xyxy). Coordinates are in **640×640 letterbox
   PIXEL space** — do not treat them as normalized.
6. **Invert the letterbox** using the recorded `r`, `padX`, `padY`:
   `x_orig = (x_letterbox − padX) / r`, `y_orig = (y_letterbox − padY) / r`. This maps boxes
   back to **original-image pixel space**. Then compose with the image→widget display transform
   so boxes overlay correctly on the (possibly fit-scaled) displayed radiograph. Clamp to image
   bounds.
7. **Per-class NMS, IoU 0.45** (NOT global). Run NMS independently within each class so that a
   caries box and an impacted-tooth box that legitimately overlap **both survive**. (Global NMS
   would wrongly suppress one of two different-class overlapping detections.)
8. Emit `Detection{ bbox (display space), label, conf }` per surviving box.

## UI
- Left to the worker's visual design. Functional must-haves only:
  - Draw each detection as a **box + class label + confidence** on the radiograph, color-coded
    per class (caries / periapical_lesion / impacted_tooth).
  - A **per-class count** (how many of each of the 3 classes were detected).
  - An **inference latency readout** (ms). Because release-build Dart `print` does not reach the
    native console, surface any needed diagnostics (per-stage timings, tensor shapes, raw first
    box) on the on-screen HUD, not via logs.
  - A **required, always-visible non-diagnostic disclaimer**: this is a research capability proof,
    **not** a diagnostic device; nothing shown is clinically validated; on-device deployment
    changes data-residency only and **does not imply or confer FDA clearance**.
  - A way to load input: gallery/file pick **and** a bundled sample radiograph for a one-tap demo.

## Platform targets
- **iOS minimum 16.6** (matches the repo's existing builds, e.g. FireDetectionYOLO), **Android
  minSdk 24**.
- The iOS **simulator is a dead end** — the vendored xcframework ships a device-only (ios-arm64)
  slice; every iteration is a signed **release** device build (debug hangs on launch on recent
  iOS/Xcode).
- Known OS traps for this model/artifact:
  - **FP32-GPU CoreML artifact can crash in Apple MPSGraph on iOS/macOS 26.3+** (SIGABRT,
    uncatchable in Dart, on the *first* inference — a fusion-pattern bug in Apple's GPU
    compiler). Not client-fixable and no `modelMode` avoids it; the durable fix is ZETIC
    filtering GPU server-side for the affected OS. Always read the **served** `target` + `apType`
    from the native console and confirm it is **not** GPU on affected OS versions.
  - **Benchmarked ≠ deployable.** The fast NPU row (median 3.71 ms) may never be served for a
    given chip; filtering GPU can drop to `TFLITE_FP16 / CPU` (~100+ ms), not the NPU. Treat
    the **served `apType` from the native console** as truth, plan for CPU-speed fallback, and
    treat the benchmark numbers above as EXPECTED, not guaranteed-served.

## Validation focus
THIS-model Tier-A correctness traps the worker must cover with tests (see VALIDATION.md Tier A).
Use the three validated demo radiographs (`demo_images/DEMO_IMAGES.md`: `val_28.png`,
`val_38.png`, `val_32.png`) as the pre/post ground-truth harness — their measured detections,
confidences, and classes are the expected outputs.
- **Registered model name `ajayshah/DentalXRayDetect` (fully-qualified `account/project`, capital
  R) ≠ folder `DentalXrayDetect` (lowercase r)** — the SDK `create(name: ...)` string must be the
  `account/project` form and match the registered case exactly; a bare project name or the folder
  name will not resolve the model.
- **Channel-major `[1,7,8400]` decode** — stride across the 8400 anchor axis, not the 7. A
  transposed read must fail the test.
- **No-extra-sigmoid score semantics** — assert scores are consumed as-is; a re-applied sigmoid
  must be caught. Its real failure mode is dynamic-range compression toward ~0.5–0.62 (see the
  Model section): a second activation crushes confidences together so the 0.45 gate no longer
  separates hits from misses — e.g. a true negative at raw 0.42 (correctly dropped) becomes
  sigmoid(0.42)=0.60 and wrongly passes, while high-confidence hits collapse toward 0.6.
- **Per-anchor max-of-3-class** — correct argmax/label assignment across `[s0,s1,s2]`.
- **Letterbox inverse round-trip** — a known box put through pre-letterbox then post-inverse must
  return to its original-image coordinates within tolerance (uses the recorded `r`, `padX`,
  `padY`).
- **Coordinate space** — boxes are 640 letterbox **pixel** space, not normalized 0–1.
- **Per-class (not global) NMS at IoU 0.45** — construct two overlapping boxes of *different*
  classes and assert **both survive**; two overlapping same-class boxes collapse to one.
- **Threshold boundary at 0.45** — a score just below 0.45 is dropped. NOTE: the implementation
  pins to the validated `demo_images/validate_demo.py` harness, which uses **strict `>`**
  (`conf > conf_thres`), so a score of exactly 0.45 is dropped — this is what reproduces the
  measured demo detections. (The earlier "≥ 0.45" prose is superseded by the harness for the
  boundary case.)
- **Honest-caveat guardrails** (data caveats to disclose, not code bugs):
  - The model is **panoramic-trained** → domain shift vs bitewing/periapical intra-oral films;
    it is unproven on those without re-validation/fine-tuning.
  - Per-class reality (measured, conf 0.45): **impacted_tooth is strong** (recall 0.82,
    precision 0.75); **caries is a screening hint, not a count** (recall ~0.35, precision 0.57);
    **periapical_lesion is data-starved** (only 9 GT instances — treat any single hit as
    anecdotal). Overall recall ~0.45. The demo images are the model's best, most legible cases,
    not representative average performance.
  - **Licenses = demo-only, non-commercial.** Model weights: **CC-BY-NC-4.0**. Demo images /
    DENTEX dataset: **CC-BY-NC-SA-4.0** (attribution: Hamamci et al., DENTEX, MICCAI 2023). Both
    non-commercial — this is a capability proof, never a shippable commercial product. The
    drop-in license-clean fallback is `Sentoz/dental-opg-cavity-detection-model` (MIT), same
    YOLO export recipe, at the cost of unverified quality.
  - **Clinical honesty:** on-device changes data-residency **only**; it does **not** confer or
    alter FDA clearance. Never overclaim in the demo.
