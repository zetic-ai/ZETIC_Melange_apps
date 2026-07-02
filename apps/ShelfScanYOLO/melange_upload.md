# Melange upload — ShelfScanYOLO

Drag these into the dashboard:
- model:  shelfscan-yolo11s-sku110k.onnx
- sample: sample_input.npy

Create the model with:
- name:    ajayshah/ShelfScanYOLO   (the SDK create(name:) must include account + project separated
                            by a slash — account `ajayshah`, project `ShelfScanYOLO`. Match case exactly.)
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  float32[1, 3, 640, 640], NCHW, values 0.0–1.0 (pixels / 255), RGB
- output tensor: float32[1, 5, 8400], channel-major; per anchor [cx, cy, w, h, object_score]
  = 4 box coords (in 640×640 letterbox pixel space) + 1 class score. NMS NOT baked in.
  Class score is already sigmoid-activated in-graph (values 0–1) — do NOT re-apply sigmoid.
- classes / labels (1): object  (a generic retail product facing / SKU on a shelf)

Then: trigger benchmark, wait for CONVERTING -> OPTIMIZING -> READY.

Paste back to the agent (it is BLOCKED until you do):
- the model name + version you registered
- the served input/output shapes the dashboard shows
- modelMode: default RUN_AUTO
  (Do NOT use RUN_ACCURACY as a crash workaround — it isn't one. The iOS/macOS
  26.3+ CoreML-GPU crash is handled server-side by ZETIC filtering the GPU
  candidate; no client mode avoids it. See CLAUDE.md section 5.)

Notes for this model (read before upload):
- Standard 640×640 YOLO11s graph (same architecture family as PyroGuard's YOLO11s), opset 12,
  static shapes — verified clean with onnx.checker and onnxruntime. Expect a clean conversion.
- Single class ("object"): the model detects generic product facings / SKUs densely (trained on
  SKU-110K). It boxes every product on a shelf — great dense-detection trade-show visual — but it
  does NOT classify brand or product category. That is by design.
- License is `other`, tied to the upstream SKU-110K dataset terms (research / D&D use). Fine for an
  on-device benchmark DEMO; flag it before any commercial productization. Apache-2.0 / MIT fallbacks
  exist (see model_selection.md) if a clean-license retail detector is ever needed.
