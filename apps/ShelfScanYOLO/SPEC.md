# SPEC: ShelfScanYOLO

> Product display name (user-facing): **ShelfSense**. Apply as display name only — iOS
> `CFBundleDisplayName`, Android `android:label`, `MaterialApp(title:)`, and the app-bar /
> loading text. Do NOT change the bundle id, the app folder name (`ShelfScanYOLO`), or the
> registered Melange model name (`ajayshah/ShelfScanYOLO`), which stay stable (CLAUDE.md §4).

## One-line pitch
Real-time, fully on-device dense retail-shelf SKU detector — draws a box on every product
facing on a store shelf (and boxes/cartons in a warehouse) with no upload. A free
on-device auto-benchmark demo for retail-execution & warehouse-CV buyers (Infilect/InfiViz,
Trax, Shopic, Arvist) across fragmented cheap Android handsets plus one edge SoC.

## Model
- **Source (HF repo / origin):** `chistopat/sku110k-yolo11-object-detector`, weights
  `weights/sku110k-yolo11-s640.pt`.
- **Architecture:** YOLO11s (Ultralytics), anchor-free single-class detector; trained on
  SKU-110K (dense retail shelves) at imgsz 640. mAP50 0.927 / mAP50-95 0.577 (repo metrics).
- **Melange model name:** `ajayshah/ShelfScanYOLO` (the SDK `create(name: ...)` must include the
  account name and project name separated by a slash — account `ajayshah`, project `ShelfScanYOLO`.
  Match case exactly. Confirmed at GATE 0).
- **Melange version:** `1` (confirmed at GATE 0).
- **Input tensor:** name `images`, `float32[1,3,640,640]`, layout **NCHW**, channel order
  **RGB**, value range **0.0–1.0** (divide pixel bytes by 255), **letterboxed** to 640×640
  (aspect preserved, padded). This is the served input tensor confirmed at GATE 0.
- **Output tensor:** name `output0`, `float32[1,5,8400]`, **channel-major per anchor**.
  For each of the 8400 anchors the 5 channels are `[cx, cy, w, h, object_score]` = 4 box
  coordinates + 1 class score. Box coordinates are in **640×640 letterbox pixel space**
  (verified raw box channels max ≈ 638–640, NOT normalized 0–1). The 8400 anchors are
  80²+40²+20² across the /8, /16, /32 strides. Confirmed at GATE 0.
- **Post-processing baked into ONNX?** **No NMS in-graph** — implement NMS in pure Dart.
  However the single **class score IS already sigmoid-activated in-graph** (verified raw
  `object_score` range 0.0–0.83, i.e. already 0–1). **Do NOT re-apply sigmoid** to the score.
  YOLO11 is anchor-free with no separate objectness channel (the 5 = 4 box coords + 1 class
  score; there is no extra objectness multiply). This is a real difference from PyroGuard —
  do not port a re-sigmoid step from a prior YOLO pipeline.
- **Classes / labels (1, verified from checkpoint `model.names`):**
  `0 → object` — a generic retail **product facing / SKU**. This model **localizes, it does
  not classify**: it does not identify brand, category, or price. The user-facing label is
  "product" / "SKU".
- **modelMode to use and why:** **RUN_AUTO** (default). No client modelMode can steer off a
  crashing artifact, and the iOS-26 GPU/MPSGraph trap is handled server-side by ZETIC
  (CLAUDE.md §5). Record the mode requested, but treat the **served artifact** (target +
  apType read from the native console) as ground truth.
- **Melange benchmark (EXPECTED, from the dashboard report — NOT a guarantee of what is
  served for a given chip; read the served row from the native console):**
  100% FP32-deployable; NPU median **5.43 ms** (low **2.83 ms**), up to **×111 vs CPU**;
  CPU median **~375 ms**; smallest memory footprint **~20 MB**; 3 quantizations available.
  Per CLAUDE.md §5, "benchmarked" ≠ "deployable" — a fast NPU row may never actually be
  served; plan for a CPU-speed fallback (hundreds of ms) until `runtimeApType=NPU` is
  confirmed on-device.

## Input source
- **A selected still image** — single-shot **image-upload** demo. Input is a photo the user
  picks from **file / gallery**, or a **bundled sample shelf photo** for a one-tap demo. There is
  **no live camera** — this is a still-image detector, not a live-feed app. Decode the selected
  image bytes to **RGB**.
- **Orientation handling — honor EXIF only.** The **one and only** orientation concern is the
  **EXIF orientation tag** on the decoded file: apply it so the image is upright before
  pre-processing. Because there is **no live camera buffer, there is NO
  camera-orientation / rotating-buffer trap** — the class of spurious-rotation bug that
  dominated PyroGuard (CLAUDE.md §5, §6 worked example) **does NOT apply here**. Do not port any
  camera-buffer WxH measurement or blind-rotate logic.
- **Displayed-image mapping is the one geometry task:** the overlay must map **letterbox pixel
  space → the on-screen displayed-image rect** correctly. Detections come back in 640×640
  letterbox coordinates; they must be un-letterboxed (see post-processing) into original-image
  pixel space and then fitted to the **displayed image's** rect (the image may be
  fit/letterboxed inside its widget, so account for that image→widget fit transform). A box that
  is geometrically correct in 640-space but drawn with the wrong displayed-image transform is
  the classic failure — verify with a known box round-trip.

## Pre-processing pipeline (ordered, exact)
1. **Load the selected image bytes; decode; apply EXIF orientation** so the image is upright
   (file / gallery pick, or the bundled sample shelf photo).
2. **Letterbox-resize to 640×640**: scale the image by a single factor `s = min(640/W,
   640/H)` to preserve aspect ratio, then center-pad the remainder to 640×640 with a constant
   gray pad (value 114 in 0–255, i.e. 0.447 after /255). **Record the exact `scale` and the
   `padX`/`padY` offsets** — the post-processing inverse must use these same numbers.
3. Ensure **RGB** channel order.
4. **Normalize** by dividing each channel by 255.0 → range 0.0–1.0.
5. **Reorder to NCHW** `[1,3,640,640]` (channel-planar: all R, then all G, then all B).
6. Flatten to a `Float32List` and wrap as `Tensor.float32List(data, shape: [1,3,640,640])`.
   Pass as the single element of the `List<Tensor>` given to `model.run(...)`, bound to input
   `images`.

## Post-processing pipeline (ordered, exact)
Coordinate spaces in play, keep them distinct: **(a)** 640×640 letterbox pixel space (raw
model output), **(b)** original-image pixel space (after inverting the letterbox),
**(c)** on-screen displayed-image space (after the image→widget fit transform).
1. Read `output0` as **channel-major `[1,5,8400]`**: iterate the **8400 anchors** and, for
   anchor `a`, read `cx=out[0*8400+a]`, `cy=out[1*8400+a]`, `w=out[2*8400+a]`,
   `h=out[3*8400+a]`, `score=out[4*8400+a]`. **Stride across the 8400, NOT across the 5** —
   the 5 is the outer (channel) dimension. Do not read 5 contiguous floats per anchor.
2. `score` (channel 4) is the single-class `object_score` and is **already sigmoid-activated
   (0–1) in-graph — use it as-is. DO NOT apply sigmoid again.**
3. **Threshold first:** keep only anchors where `score > conf_threshold` (default **0.25**).
   Threshold BEFORE any box geometry to avoid decoding 8400 boxes needlessly.
4. **`cxcywh → x1y1x2y2`** in 640-space: `x1=cx-w/2, y1=cy-h/2, x2=cx+w/2, y2=cy+h/2`
   (still in 640×640 letterbox pixel space).
5. **Invert the letterbox** — the **exact reverse** of pre-processing step 2, using the
   recorded `scale`, `padX`, `padY`: `x_orig = (x_letterbox - padX) / scale`,
   `y_orig = (y_letterbox - padY) / scale`, applied to both corners. This yields boxes in the
   original-image pixel space. (Then apply the image→widget fit transform to reach the on-screen
   displayed-image space when drawing — see Input source.)
6. **Single-class NMS, IoU 0.45.** One class → **global NMS is correct** (no per-class
   grouping needed). Dense shelves pack facings tightly, so the IoU threshold is sensitive:
   too aggressive **merges adjacent SKUs** (under-counts), too loose **double-counts one
   product**. 0.45 is the validated default (see demo images); expose it so a live app can
   tune.
7. Emit `Detection{ bbox, conf }` — single label "product" / "SKU". The count of surviving
   detections is the headline retail metric.

## UI
Left to the worker (visual design is the worker's choice). **Functional must-haves only:**
- **Display the uploaded shelf image** with detection boxes and per-box confidence overlaid,
  drawn correctly mapped from letterbox → the displayed-image rect (image→widget fit).
- **Detection count** — a prominent "N products detected" readout (the headline retail
  number; the dense single-class nature makes a large count the money shot).
- **Inference latency readout** (ms).
- **A way to load input** — file / gallery pick, plus a **bundled sample shelf image** for a
  one-tap demo.
- Per CLAUDE.md §5 (release builds swallow Dart `print`), surface any diagnostics you need —
  per-stage timings, tensor shape, the raw first detection box, the recorded `scale`/`padX`/
  `padY` — on an on-screen **debug/HUD line**, not via `print`.

## Platform targets
- **iOS minimum 16.6** (matches the FireDetectionYOLO `IPHONEOS_DEPLOYMENT_TARGET`),
  **Android minSdk 24** (repo convention / PyroGuard worked example).
- **Known OS traps for this model/artifact:**
  - **iOS/macOS 26.3+ MPSGraph GPU crash:** an FP32-GPU CoreML artifact can load cleanly
    ("BackendSelectionExecutor: success") then **SIGABRT at the first inference** inside
    Apple's MPSGraph compiler (`MLIR pass manager failed`) — an Apple GPU-compiler bug hit by
    standard YOLO/ViT-style attention heads, uncatchable in Dart. **Not client-fixable:** no
    modelMode steers off it; the durable fix is ZETIC filtering the GPU candidate server-side
    for the affected OS. If you hit it on a new OS, escalate to ZETIC. Always read the
    **served** target + apType from the native console and confirm it is **not GPU** on
    affected OS versions.
  - **The served artifact is truth, not the benchmark row.** Read `apType` / target from the
    native console — that is ground truth, not the dashboard's headline NPU number. Expect a
    realistic non-crashing fallback of **TFLITE_FP16 / CPU** (hundreds of ms), NOT
    necessarily the NPU; "removed the crash" and "got NPU speed" are two separate wins.
  - **iOS simulator is a dead end** (device-only ios-arm64 slice for the Melange native
    library) — every iteration is a signed **release** device build (debug hangs on launch on
    recent iOS/Xcode).
  - **If too slow on the target device:** drop-in lighter same-repo option
    `weights/sku110k-yolo11-n640.pt` (YOLO11n) at the identical 640 shape / 1 class /
    `[1,5,8400]` output — no pipeline changes needed.

## Validation focus
The specific THIS-model correctness traps the worker must cover with Tier-A tests
(VALIDATION.md). Use the **demo images as the pre/post-processing ground-truth harness** —
`demo_images/` ships 3 validated SKU-110K frames scored against measured ONNX output
(pipeline + scoring in `validate_demo.py`), with **median recall 0.903** across 36 test
shelves (@IoU≥0.5). The Dart pipeline must reproduce these numbers on the same inputs.
- **Channel-major `[1,5,8400]` decode** — stride across the 8400 anchors, not across the 5.
  This is the single easiest trap and silently corrupts every box if wrong.
- **No-extra-sigmoid score semantics** — assert the class score is consumed **as-is** and NOT
  passed through a second sigmoid. This is a **real difference from PyroGuard** (whose scores
  were not pre-activated the same way); verify a known raw score maps straight to a known
  confidence, and that thresholding at 0.25 selects the same anchors the demo harness does.
- **Letterbox inverse round-trip (to displayed-image space)** — take a known box in 640-space,
  invert with the recorded `scale`/`padX`/`padY` to the correct original-image pixel location,
  then apply the image→widget fit transform and assert it lands on the correct on-screen
  displayed-image location. A hard-coded aspect or a mismatched pad/scale silently misplaces
  every box.
- **Single-class (global) NMS, IoU 0.45** — assert global NMS (not per-class) and, on a
  dense demo frame, that the threshold neither **merges adjacent facings** (under-count) nor
  **double-counts** one product; the surviving count should track the demo-image detected
  counts (491 / 221 / 159 for the three demo frames).
- **Coordinate space** — assert boxes are treated as **640 pixel-space**, not normalized 0–1
  (raw box channels reach ~640, not ~1).
- **Threshold boundary** — assert the `> conf_threshold` comparison behaves correctly at the
  0.25 boundary (inclusive/exclusive consistency) so recall/precision match the demo harness.

## Licensing — FLAG (demo-only; clear before any commercial use)
- **Model:** license is `other`, tied to the upstream **SKU-110K dataset terms
  (research / non-commercial "D&D" use)**; the model card says to keep downstream use aligned
  with SKU-110K terms. For ZETIC's use — an **on-device inference benchmark demo**, not a
  shipped retail-analytics product on customer data — this is demo/research use and is
  acceptable, but it is a **real GTM flag if productized**. Clean-license fallbacks if that
  day comes: `prince4332/yolov26-product-detection` (Apache-2.0, YOLO26 — accept conversion
  risk) or `hatuankiet/YOLOv12S_SKU110K` (MIT, YOLOv12). AGPL-3.0 (foduu) is deliberately
  avoided as the worst license for a proprietary app.
- **Demo images:** the `demo_images/` frames are from **SKU-110K (Goldman et al., CVPR 2019)**,
  distributed for **academic / non-commercial** use (mirror tags `cc-by-nc-2.0`). Fine for an
  internal/research benchmark demo; **NOT cleared for commercial redistribution**. Replace
  with owned/permissively-licensed shelf photos before any commercial context.
