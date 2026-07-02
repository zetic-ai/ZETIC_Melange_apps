# Melange upload — RetinaDRGrade

Drag these into the dashboard:
- model:  vit-base-dr-grade.onnx
- sample: sample_input.npy

Create the model with:
- name:    ajayshah/RetinaDRGrade
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  float32[1,3,224,224], NCHW, RGB.
  Preprocessing baked into the app pipeline (NOT the ONNX): resize 224x224 bilinear,
  /255, then normalize mean=[0.5,0.5,0.5] std=[0.5,0.5,0.5] (pixels -> [-1,1]).
- output tensor: float32[1,5], RAW LOGITS (no softmax baked in).
  Apply softmax + argmax downstream. argmax index == canonical DR grade directly
  (id2label is the identity map {0:0,1:1,2:2,3:3,4:4}).
- classes / labels: [0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative].
  Referable = grade >= 2.

Note (size / on-device): the ONNX is ~343 MB fp32 (ViT-base). Treat this as a
first-launch download and on-device footprint consideration — the model must be
pulled and cached on device before the first inference. (Melange decides serving
precision server-side; do not pre-quantize the ONNX.)

Then: trigger benchmark, wait for CONVERTING -> OPTIMIZING -> READY.

Paste back to the agent (it is BLOCKED until you do):
- the model name + version you registered
- the served input/output shapes the dashboard shows
- modelMode: default RUN_AUTO
  (Do NOT use RUN_ACCURACY as a crash workaround — it isn't one. The iOS/macOS
  26.3+ CoreML-GPU crash is handled server-side by ZETIC filtering the GPU
  candidate; no client mode avoids it. See CLAUDE.md section 5.)
