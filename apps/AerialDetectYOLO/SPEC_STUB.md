# SPEC: AerialDetectYOLO

## One-line pitch
Real-time on-device aerial/drone object detector (people + vehicles seen top-down) for industrial
inspection and agriculture surveillance prospects.

## Model
- Source (HF repo / origin): ENOT-AutoDL/yolov8s_visdrone, weights `baseline_enot/weights/best.pt`
- Architecture: YOLOv8s (Ultralytics), anchor-free; fine-tuned on VisDrone-2019 at imgsz 928
- Melange model name: <<FILL AT GATE 0>>   (requested: ajayshah/AerialDetectYOLO)
- Melange version: <<FILL AT GATE 0>>       (requested: 1)
- Input tensor: float32[1,3,928,928], NCHW, value range 0.0–1.0 (divide pixels by 255), RGB order
  - Served input shape (echo from dashboard): <<FILL AT GATE 0>>
- Output tensor: float32[1,14,17661], channel-major. Per anchor: [cx, cy, w, h, p0..p9] =
  4 box coords (in 928×928 letterbox space) + 10 raw class scores. 17661 anchors = 116²+58²+29²
  across /8, /16, /32 strides.
  - Served output shape (echo from dashboard): <<FILL AT GATE 0>>
- Post-processing baked into ONNX? No. No NMS, no sigmoid in-graph (raw YOLOv8 head). YOLOv8 has
  no separate objectness channel — the 14 = 4 box + 10 class. Threshold on max class score in Dart.
- Classes / labels (10, verified from checkpoint model.names, VisDrone order):
  0 pedestrian, 1 people, 2 bicycle, 3 car, 4 van, 5 truck, 6 tricycle, 7 awning-tricycle,
  8 bus, 9 motor
  (Note: VisDrone splits "pedestrian" = person walking/standing vs "people" = person in other
  poses; both are humans. "motor" = motorcycle/scooter.)
- modelMode to use and why: RUN_AUTO (default). No client mode steers off a crashing artifact;
  the iOS-26 GPU/MPSGraph trap is handled server-side by ZETIC. See CLAUDE.md §5.

## Input source
- Rear camera (live demo), cheapest usable pixel format (iOS BGRA / Android YUV420)
- Pixel format or sample rate requested: cheapest usable camera format; convert to RGB in Dart
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
3. Keep anchors where max class score > threshold (start 0.25). Threshold BEFORE box geometry.
4. Class scores are already 0–1 (no sigmoid needed) — confirm on first real run; do NOT double-activate.
5. cxcywh → x1y1x2y2 (coords are in 928×928 letterbox space).
6. Undo letterbox (exact reverse of pre-processing order) into screen space.
7. Per-class NMS, IoU 0.45 (per-class, NOT global — two different-class boxes may overlap).
8. Emit Detection{bbox, label, conf}.

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
- Channel-major [1,14,17661] decode (stride across anchors, not the 14) — easiest trap.
- Letterbox inverse round-trip at 928 (not 640) — a hard-coded 640 will silently misplace boxes.
- Score semantics: confirm class scores need NO sigmoid for this checkpoint (don't double-activate).
- Per-class (not global) NMS.
- Coordinate space: pixel-space 928, not normalized 0–1.
- Orientation: verify real camera buffer WxH on-device; assert the chosen transform round-trips a
  known box. Small aerial objects make a transpose bug easy to miss.
