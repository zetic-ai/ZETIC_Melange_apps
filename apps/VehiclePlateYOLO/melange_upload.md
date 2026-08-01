# Melange upload — VehiclePlateYOLO

Drag these into the dashboard:
- model:  koushim-yolov8-license-plate.onnx
- sample: sample_input.npy

Create the model with:
- name:    ajayshah/VehiclePlateYOLO
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  float32[1,3,640,640], NCHW (RGB, values 0.0-1.0 / divide by 255)
- output tensor: float32[1,5,8400], channel-major; per anchor [cx, cy, w, h, plate_conf]
                 (coords in 640x640 letterboxed space; 8400 anchors; NMS NOT baked in)
- classes / labels: ["license_plate"]  (single class)

Then: trigger benchmark, wait for CONVERTING -> OPTIMIZING -> READY.

Paste back to the agent (it is BLOCKED until you do):
- the model name + version you registered
- the served input/output shapes the dashboard shows
- modelMode: default RUN_AUTO
  (Do NOT use RUN_ACCURACY as a crash workaround - it isn't one. The iOS/macOS
  26.3+ CoreML-GPU crash is handled server-side by ZETIC filtering the GPU
  candidate; no client mode avoids it. See CLAUDE.md section 5.)
