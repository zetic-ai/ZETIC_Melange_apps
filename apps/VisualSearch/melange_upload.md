# Melange upload — VisualSearch (TWO models)

This app is a **two-model on-device visual-search pipeline**: a YOLO11n detector
localizes the salient product/object, the crop is embedded by a MobileCLIP2-S0 image
tower, and ONLY the resulting 512-d vector would go to a server for catalog matching.
Upload BOTH models as **separate** Melange models. Do the two uploads independently;
the app's name/version injection is BLOCKED at GATE 0 until you paste back the
registered names/versions + served shapes for **both** (the worker build runs in
parallel and needs neither to proceed).

Both ONNX files are **fully STATIC** (zero dynamic axes — verified programmatically in
`export.py`) and FP32 (Melange owns precision — no fp16 baked in). Ops are all
standard (Conv/Concat/Mul/Sigmoid/Split/MaxPool for the detector; Conv/MatMul/Softmax/
LayerNorm/Add for the embedding tower), so both should sail through CONVERTING.

---

## Model 1 of 2 — DETECTOR (salient object)

Drag these into the dashboard:
- model:  `visualsearch_detect.onnx`     (10.7 MB, FP32, opset 12 — Ultralytics YOLO11n, COCO)
- sample: `sample_input_detect.npy`      (float32, shape [1, 3, 640, 640])

Create the model with:
- name:    `ajayshah/VisualSearchDetect`
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  `images` — float32[1, 3, 640, 640], NCHW, RGB, letterboxed, values 0..1 (pixel/255)
- output tensor: `output0` — float32[1, 84, 8400], channel-major; per anchor
                 [cx, cy, w, h, + 80 COCO class scores]; coords in 640×640 letterbox space
- post-processing baked in? **NO.** Threshold + cxcywh→xyxy + un-letterbox + NMS are pure-Dart.
- classes / labels: 80 COCO classes (the app uses the box only as a salient-object
  localizer — it crops the top box; the class label is incidental).

Then: trigger benchmark, wait for CONVERTING → OPTIMIZING → READY.

---

## Model 2 of 2 — EMBEDDING (image tower only)

Drag these into the dashboard:
- model:  `visualsearch_embed.onnx`      (45.8 MB, FP32, opset 14 — MobileCLIP2-S0 image tower)
- sample: `sample_input_embed.npy`       (float32, shape [1, 3, 256, 256])

Create the model with:
- name:    `ajayshah/VisualSearchEmbed`
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  `image` — float32[1, 3, 256, 256], NCHW, RGB, values 0..1 (pixel/255),
                 **NO ImageNet mean/std** (MobileCLIP uses plain [0,1], mean 0 / std 1)
- output tensor: `embedding` — float32[1, 512], **L2-normalized in-graph** (unit vector)
- post-processing baked in? **YES for the embedding** — the CLIP projection AND the
  L2-normalization are inside the ONNX graph, so cosine similarity == a plain dot
  product of two outputs. No Dart post-processing beyond the dot product.
- classes / labels: N/A — this is a 512-d feature vector, not a classifier.

Then: trigger benchmark, wait for CONVERTING → OPTIMIZING → READY.

---

## Paste back to the agent (unblocks name/version injection — for BOTH models)

For **VisualSearchDetect**:
- registered model name + version (expected `ajayshah/VisualSearchDetect` v1)
- served input/output shapes the dashboard shows (confirm [1,3,640,640] → [1,84,8400])
- served `runtimeApType` (NPU / GPU / CPU) from the device console, if known

For **VisualSearchEmbed**:
- registered model name + version (expected `ajayshah/VisualSearchEmbed` v1)
- served input/output shapes the dashboard shows (confirm [1,3,256,256] → [1,512])
- served `runtimeApType` (NPU / GPU / CPU) from the device console, if known

modelMode for both: default **RUN_AUTO**.
- Do **NOT** use RUN_ACCURACY as a crash workaround — it is not one. The iOS/macOS 26.3+
  CoreML-GPU MPSGraph crash is handled **server-side** by ZETIC filtering the GPU candidate
  for the affected OS; no client modelMode avoids it (all four returned the same crashing
  artifact on PyroGuard). See CLAUDE.md section 5. Note: the embedding tower is a FastViT
  with self-attention heads — the exact ViT-style attention family that hit the iOS-26 GPU
  bug — so watch the served apType on iOS 26.3+ and escalate to ZETIC to filter GPU if it
  crashes at first inference.
- "Benchmarked" ≠ "served": a fast NPU row in the report may never be served for a given
  chip. Read the *served* target+apType from the native console — that is ground truth.
