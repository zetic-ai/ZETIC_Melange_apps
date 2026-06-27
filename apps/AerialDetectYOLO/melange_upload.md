# Melange upload — AerialDetectYOLO

Drag these into the dashboard:
- model:  aerialdetect-yolov8s-visdrone.onnx
- sample: sample_input.npy

Create the model with:
- name:    ajayshah/AerialDetectYOLO
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  float32[1, 3, 928, 928], NCHW, values 0.0–1.0 (pixels / 255), RGB
- output tensor: float32[1, 14, 17661], channel-major; per anchor [cx, cy, w, h, p0..p9]
  = 4 box coords (in 928×928 letterbox space) + 10 class scores. NMS NOT baked in.
- classes / labels (10, VisDrone order):
  pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor

Then: trigger benchmark, wait for CONVERTING -> OPTIMIZING -> READY.

Paste back to the agent (it is BLOCKED until you do):
- the model name + version you registered
- the served input/output shapes the dashboard shows
- modelMode: default RUN_AUTO
  (Do NOT use RUN_ACCURACY as a crash workaround — it isn't one. The iOS/macOS
  26.3+ CoreML-GPU crash is handled server-side by ZETIC filtering the GPU
  candidate; no client mode avoids it. See CLAUDE.md section 5.)

Notes for this model (read before upload):
- Input is 928×928, NOT the usual 640×640. This is deliberate: the model is aerial/drone-trained
  (VisDrone) at imgsz 928 so it can resolve small top-down objects. Keep the shape exactly as
  exported; it is static. Heavier input = expect higher on-device latency than a 640 model
  (a Tier C runtime concern, not a conversion issue).
- License is Apache-2.0 (ENOT-AutoDL/yolov8s_visdrone) — clean for the GTM demo.
