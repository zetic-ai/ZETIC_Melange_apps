# SPEC: ShelfScanYOLO

## One-line pitch
Real-time, fully on-device dense retail-shelf SKU detector — boxes every product facing on a store
shelf (and boxes/cartons in a warehouse) with no upload — a free auto-benchmark demo for retail
execution & warehouse CV buyers (Infilect/InfiViz, Trax, Shopic, Arvist) across cheap Android
handsets + one edge SoC.

## Model
- Source (HF repo / origin): chistopat/sku110k-yolo11-object-detector, weights `weights/sku110k-yolo11-s640.pt`
- Architecture: YOLO11s (Ultralytics), anchor-free; trained on SKU-110K (dense retail shelves) at imgsz 640
- Melange model name: <<FILL AT GATE 0>>   (requested: ShelfScanYOLO — registered under ZETIC org, match case exactly)
- Melange version: <<FILL AT GATE 0>>       (requested: 1)
- Input tensor: float32[1,3,640,640], NCHW, value range 0.0–1.0 (divide pixels by 255), RGB order
  - Served input shape (echo from dashboard): <<FILL AT GATE 0>>
- Output tensor: float32[1,5,8400], channel-major. Per anchor: [cx, cy, w, h, object_score] =
  4 box coords (in 640×640 letterbox pixel space) + 1 class score. 8400 anchors = 80²+40²+20²
  across the /8, /16, /32 strides.
  - Served output shape (echo from dashboard): <<FILL AT GATE 0>>
- Post-processing baked into ONNX? No. No NMS in-graph. The single class score IS already
  sigmoid-activated in-graph (verified raw-output range 0.0–1.0) — do NOT re-apply sigmoid. YOLO11
  is anchor-free with no separate objectness channel (5 = 4 box + 1 class). Threshold on the class
  score in Dart.
- Classes / labels (1, verified from checkpoint model.names):
  0 object  (a generic retail product facing / SKU; NOT brand/category — it localizes, does not classify)
- modelMode to use and why: RUN_AUTO (default). No client mode steers off a crashing artifact; the
  iOS-26 GPU/MPSGraph trap is handled server-side by ZETIC. See CLAUDE.md §5.

## Input source
- Rear camera (live demo), cheapest usable pixel format (iOS BGRA / Android YUV420)
- Pixel format or sample rate requested: cheapest usable camera format; convert to RGB in Dart
- Orientation handling required: MEASURE the real buffer WxH on-device before assuming rotation.
  On the PyroGuard iOS setup the buffer arrived upright (720×1280) and the bug was a *spurious*
  overlay rotation — do not blind-rotate. Verify per device/format.

## Pre-processing pipeline (ordered, exact)
1. Capture frame bytes.
2. Letterbox-resize to 640×640 (pad value 114-gray / 0.5, preserve aspect ratio).
3. Convert to RGB channel order if source is BGRA/YUV.
4. Normalize /255.0 → 0.0–1.0.
5. Reorder to NCHW [1,3,640,640].
6. Flatten to Float32List, wrap as Tensor.float32List(data, shape: [1,3,640,640]).

## Post-processing pipeline (ordered, exact)
1. Read output as channel-major [1,5,8400]: stride across the 8400 anchors, NOT across the 5.
2. For each anchor, the class score is channel 4 (single class). Score is already 0–1 (sigmoid
   baked in) — do NOT double-activate.
3. Keep anchors where score > threshold (start 0.25). Threshold BEFORE box geometry.
4. cxcywh → x1y1x2y2 (coords are in 640×640 letterbox pixel space).
5. Undo letterbox (exact reverse of pre-processing order) into screen space.
6. NMS, IoU 0.45. (Single class → global NMS is fine; dense shelves pack boxes tightly, so tune the
   IoU threshold — too aggressive merges adjacent SKUs, too loose double-counts.)
7. Emit Detection{bbox, conf} (single label "object" / "SKU").

## UI
- Left to the worker. Functional must-haves: live overlay of boxes + confidence, a live SKU/product
  count (the headline retail metric), inference latency readout. Consider a face-count-style large
  "N products detected" number given the dense single-class nature.

## Platform targets
- iOS minimum 16.6+, Android minSdk 24.
- Known OS traps for this model/artifact:
  - FP32-GPU CoreML artifact can crash in Apple MPSGraph on iOS/macOS 26.3+ (not client-fixable;
    ZETIC filters GPU server-side). Read the *served* target+apType from the native console.
  - Realistic non-crashing fallback is often TFLITE_FP16/CPU (hundreds of ms), not the NPU. If too
    slow, drop-in lighter same-repo option: `weights/sku110k-yolo11-n640.pt` (YOLO11n) at the same
    640 shape / 1 class.

## Validation focus
- Channel-major [1,5,8400] decode (stride across anchors, not the 5) — easiest trap.
- Score semantics: the class score is ALREADY sigmoid-activated in-graph — assert no double sigmoid.
- Letterbox inverse round-trip at 640 — a hard-coded aspect assumption will silently misplace boxes.
- NMS tuning for DENSE scenes: SKU-110K shelves are packed; verify IoU threshold neither merges
  adjacent facings nor double-counts one product.
- Coordinate space: pixel-space 640, not normalized 0–1.
- Orientation: verify real camera buffer WxH on-device; assert the chosen transform round-trips a
  known box.
