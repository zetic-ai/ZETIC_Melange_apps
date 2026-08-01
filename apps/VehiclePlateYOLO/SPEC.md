# SPEC: VehiclePlateYOLO  (FINALIZED — GATE 1)

## One-line pitch
Real-time on-device license-plate detector for smart-city / parking-management
prospects (street & parking-lot cameras) — single YOLO forward pass, no OCR, no
two-stage vehicle-crop chain.

## Model
- Source (HF repo / origin): Koushim/yolov8-license-plate-detection (MIT)
  (weight file `best.pt`; exported ONNX `koushim-yolov8-license-plate.onnx`)
- Architecture: YOLOv8n fine-tuned, single-stage, single-class plate detector
  (~12 MB ONNX). Selected at GATE 0 for MIT (GTM-clean) over the AGPL technical
  pick morsetechlab/yolov11-license-plate-detection.
- Melange model name: **ajayshah/VehiclePlateYOLO**
- Melange version: **1** (status: READY)
- Input tensor: float32[1,3,640,640], NCHW, values 0.0-1.0 (divide by 255), RGB
  - **Served input shape: float32[1,3,640,640]** (dashboard-confirmed)
- Output tensor: float32[1,5,8400], channel-major; per anchor [cx, cy, w, h, plate_conf];
  coords in 640x640 letterboxed space; 8400 anchors over 80/40/20 grids
  - **Served output shape: float32[1,5,8400]** (dashboard-confirmed)
- Post-processing baked into ONNX? No. NMS is NOT baked in — implement in pure Dart.
  **Sigmoid IS baked in** (confirmed Sigmoid node in graph) — do NOT re-apply sigmoid to
  plate_conf in the decode.
- Classes / labels: ["license_plate"] (single class, lowercase as exported)
- modelMode: **RUN_AUTO**. Dashboard-confirmed served path: **NPU, ~1.33 ms low**. Per
  CLAUDE.md §5 the served artifact is on-device ground truth — read `runtimeApType` from the
  native console; don't assume the dashboard number is what runs.

## Input source
- Rear camera (street / parking scene), cheapest usable pixel format
  (BGRA on iOS, YUV420 on Android — convert to RGB)
- Orientation handling required: measure the real buffer WxH on-device; on the
  PyroGuard iOS setup the BGRA buffer arrived UPRIGHT (720x1280) needing NO rotation.
  Do not assume landscape — verify, then map boxes accordingly.

## Pre-processing pipeline (ordered, exact)
1. Capture frame bytes
2. Letterbox-resize to 640x640 (pad 0.5), preserving aspect
3. Convert source pixel format -> RGB (drop alpha; BGR->RGB if needed)
4. Normalize /255.0 to 0.0-1.0
5. Reorder to NCHW [1,3,640,640]
6. Flatten to Float32List, wrap as Tensor.float32List(data, shape:[1,3,640,640])

## Post-processing pipeline (ordered, exact)
1. Read output0 as channel-major [1,5,8400]: stride across the 8400 anchors, NOT
   across the 5 — channel c, anchor a lives at index c*8400 + a.
2. For each anchor read [cx, cy, w, h, plate_conf].
3. Confidence is already activated (Sigmoid baked into the exported graph) — apply NO
   extra sigmoid in Dart. Still assert against the real tensor in Tier A.
4. Keep anchors where plate_conf > threshold (default 0.25). Threshold BEFORE
   computing box geometry so rejected anchors are nearly free.
5. cxcywh -> x1y1x2y2.
6. Undo letterbox (exact reverse of pre-processing) into screen space.
7. NMS (single class -> global NMS is fine here), IoU 0.45.
8. Emit Detection{bbox, label:"license_plate", conf}.

## UI
- Left to the worker. Functional must-haves: live overlay of plate boxes with
  confidence, live plate count, inference-latency readout.

## Platform targets
- iOS 16.6+, Android minSdk 24
- Known OS traps: FP32-GPU CoreML artifact can crash in MPSGraph on iOS/macOS 26.3+;
  not client-fixable (no modelMode avoids it) — read the SERVED target+apType from the
  native console and confirm it is not GPU on affected OS versions. (Current served path is NPU.)

## Validation focus (Tier A traps most likely for THIS model)
- Tensor layout: [1,5,8400] is channel-major — decode against a hand-built tensor with
  exactly one known plate box; assert stride-across-anchors (the #1 silent-wrong trap).
- Letterbox inverse round-trip: forward letterbox then inverse returns a known box
  within tolerance.
- Coordinate space: outputs are in 640x640 pixel space (cx,cy,w,h), not normalized.
- Score semantics: confirm plate_conf needs NO extra sigmoid (baked in) — test against
  the real tensor.
- Threshold boundary: just-below dropped, just-above kept.
- Suppression: single class, so global NMS; still test two overlapping boxes collapse
  to one and a distant box survives.
- Orientation: assert the chosen transform round-trips a known box for the buffer
  orientation actually measured on-device.

## License
- MIT (Koushim). GTM-clean. Provenance note: weights trained with AGPL Ultralytics tooling;
  confirm before any productization claim. Lower-recall risk on angled/small/distant plates vs the
  AGPL YOLO11s alternative (nano model) — acceptable for the demo.
