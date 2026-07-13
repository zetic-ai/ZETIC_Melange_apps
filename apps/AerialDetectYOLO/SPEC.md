# SPEC: AerialDetectYOLO  (FINALIZED — GATE 1)

## One-line pitch
Real-time on-device aerial/drone object detector (people + vehicles seen top-down) for industrial
inspection and agriculture surveillance prospects.

## Model
- Source (HF repo / origin): ENOT-AutoDL/yolov8s_visdrone, weights `baseline_enot/weights/best.pt`
- Architecture: YOLOv8s (Ultralytics-compatible), anchor-free; fine-tuned on VisDrone-2019 at imgsz 928
- Melange model name: **ajayshah/AerialDetectYOLO**
- Melange version: **1** (status: OPTIMIZING at GATE-0 paste-back — shapes confirmed; see lifecycle note)
- Input tensor: float32[1,3,928,928], NCHW, value range 0.0–1.0 (divide pixels by 255), RGB order
  - **Served input shape: float32[1,3,928,928]** (dashboard-confirmed; note 928, NOT 640)
- Output tensor: float32[1,14,17661], channel-major. Per anchor: [cx, cy, w, h, p0..p9] =
  4 box coords (in 928×928 letterbox space) + 10 class scores. 17661 anchors = 116²+58²+29²
  across /8, /16, /32 strides.
  - **Served output shape: float32[1,14,17661]** (dashboard-confirmed)
- Post-processing baked into ONNX? **No NMS.** Threshold on max class score in Dart. YOLOv8 has no
  separate objectness channel — the 14 = 4 box + 10 class.
- **⚠ SIGMOID — MUST RESOLVE IN TIER A (cross-app discrepancy):** the two sibling Ultralytics
  exports in this batch (RetailShelf, VehiclePlate) have a Sigmoid baked into the graph, but the
  Explorer reported the **ENOT export has NO sigmoid baked** (raw YOLOv8 head). These cannot both be
  treated the same. If no sigmoid is baked, the class channels are **logits** (not 0–1) and Dart
  MUST apply sigmoid before thresholding; thresholding raw logits at 0.25 would silently pass almost
  everything. The worker must determine this definitively before relying on any threshold:
  (a) inspect the exported graph for a Sigmoid node on the class channels, AND
  (b) check the value range of a real output tensor (logits exceed [0,1]; activated scores don't).
  Do NOT assume — this is the highest-risk trap in this app.
- Classes / labels (10, verified from checkpoint model.names, VisDrone order):
  0 pedestrian, 1 people, 2 bicycle, 3 car, 4 van, 5 truck, 6 tricycle, 7 awning-tricycle,
  8 bus, 9 motor
  (VisDrone splits "pedestrian" = person walking/standing vs "people" = person in other poses;
  both are humans. "motor" = motorcycle/scooter. Do not collapse blindly.)
- modelMode: **RUN_AUTO** (default). Per CLAUDE.md §5 the served artifact is on-device ground truth.
  928×928 is ~2.1× the compute of 640 → if AUTO does not land on NPU here, expect a heavier
  CPU-fallback latency than the two 640 apps.

## Model lifecycle note (GATE-0 status)
At paste-back the Melange model was **still OPTIMIZING** (not yet READY). The human explicitly
build-unblocked this app on the confirmed shapes: the worker MAY build the full Flutter app and run
the entire Tier A unit/benchmark battery now against the served shapes above. The worker must PARK
only at the live `model.run()` step (real on-device inference) until the model reaches READY — do
not block app construction or pure-Dart tests on it.

## Input source
- Rear camera (live demo), cheapest usable pixel format (iOS BGRA / Android YUV420); convert to RGB.
- Orientation handling required: MEASURE the real buffer WxH on-device before assuming rotation.
  On the PyroGuard iOS setup the buffer arrived upright (720×1280) and the bug was a *spurious*
  overlay rotation — do not blind-rotate. Verify per device/format.

## Pre-processing pipeline (ordered, exact)
1. Capture frame bytes.
2. Letterbox-resize to 928×928 (pad value 0.5/114-gray, preserve aspect ratio).
3. Convert to RGB channel order if source is BGRA/YUV.
4. Normalize /255.0 → 0.0–1.0.
5. Reorder to NCHW [1,3,928,928].
6. Flatten to Float32List, wrap as Tensor.float32List(data, shape: [1,3,928,928]).

## Post-processing pipeline (ordered, exact)
1. Read output as channel-major [1,14,17661]: stride across the 17661 anchors, NOT across the 14.
2. For each anchor, class scores are channels 4..13; take max class score + its index.
3. Apply sigmoid to class scores IFF Tier A confirms it is not baked in (see ⚠ above). Then keep
   anchors where activated max class score > threshold (start 0.25). Threshold BEFORE box geometry.
4. cxcywh → x1y1x2y2 (coords are in 928×928 letterbox space).
5. Undo letterbox (exact reverse of pre-processing order) into screen space.
6. Per-class NMS, IoU 0.45 (per-class, NOT global — two different-class boxes may overlap).
7. Emit Detection{bbox, label, conf}.

## UI
- Left to the worker. Functional must-haves: live overlay of boxes + confidence, per-class live
  count (10 VisDrone classes), inference latency readout.

## Platform targets
- iOS minimum 16.6+, Android minSdk 24.
- Known OS traps for this model/artifact:
  - FP32-GPU CoreML artifact can crash in Apple MPSGraph on iOS/macOS 26.3+ (not client-fixable;
    ZETIC filters GPU server-side). Read the *served* target+apType from the native console.
  - 928×928 input is ~2.1× the compute of 640×640 → expect higher latency; realistic non-crashing
    fallback is often TFLITE_FP16/CPU (hundreds of ms), not the NPU. If too slow, drop-in lighter
    same-license option: ENOT-AutoDL/yolov8s_visdrone `baseline_ultralytics/weights/best.pt`
    re-exported at imgsz 640 (identical 10 classes).

## Validation focus
- **Sigmoid-or-not (highest risk):** resolve the ⚠ discrepancy above with a graph check + value-range
  check before trusting any threshold.
- Channel-major [1,14,17661] decode (stride across anchors, not the 14) — easiest trap.
- Letterbox inverse round-trip at **928** (not 640) — a hard-coded 640 will silently misplace boxes.
- Per-class (not global) NMS.
- Coordinate space: pixel-space 928, not normalized 0–1.
- Orientation: verify real camera buffer WxH on-device; assert the chosen transform round-trips a
  known box. Small aerial objects make a transpose bug easy to miss.

## License
- Apache-2.0 (ENOT-AutoDL/yolov8s_visdrone) — GTM-clean. Genuinely aerial-trained (VisDrone), not a
  COCO substitute.
