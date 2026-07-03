# Melange upload — SafetyPPEYOLO

Drag these into the dashboard:
- model:  safetyppe-8s.onnx
- sample: sample_input.npy

Create the model with:
- name:    ajayshah/SafetyPPEYOLO
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  float32[1,3,640,640], NCHW, values 0.0-1.0
- output tensor: float32[1,17,8400] — per anchor [cx, cy, w, h, 13 class scores],
  channel-major, coords in 640x640 letterbox space, scores already sigmoid-applied,
  NMS NOT baked in
- classes / labels (id order 0-12):
  Fall-Detected, Gloves, Goggles, Hardhat, Mask, NO-Gloves, NO-Goggles,
  NO-Hardhat, NO-Mask, NO-Safety Vest, No_Harness, Person, Safety Vest

Then: trigger benchmark, wait for CONVERTING -> OPTIMIZING -> READY.

Paste back to the agent (it is BLOCKED until you do):
- the model name + version you registered
- the served input/output shapes the dashboard shows
- modelMode: default RUN_AUTO
  (Do NOT use RUN_ACCURACY as a crash workaround — it isn't one. The iOS/macOS
  26.3+ CoreML-GPU crash is handled server-side by ZETIC filtering the GPU
  candidate; no client mode avoids it. See CLAUDE.md section 5.)
