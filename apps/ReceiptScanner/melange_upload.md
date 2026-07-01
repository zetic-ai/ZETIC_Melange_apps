# Melange upload — ReceiptScanner

This app uses **TWO Melange models** (detector + recognizer). Upload BOTH as separate
models. Each has its own ONNX + sample_input. (Rationale for two models is in
model_selection.md: a single recognizer cannot localize the many text lines on a receipt,
and the only single-model end-to-end OCRs are autoregressive/dynamic → not Melange-fit.)

---

## Model 1 of 2 — DETECTOR (finds text-line boxes)

Drag these into the dashboard:
- model:  ppocrv5_mobile_det.onnx
- sample: ppocrv5_mobile_det_sample_input.npy

Create the model with:
- name:    ajayshah/ReceiptScannerDet
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  float32[1,3,960,640], NCHW
                 (preprocess: resize/letterbox to 960x640, /255, ImageNet mean
                  [0.485,0.456,0.406] std [0.229,0.224,0.225], channel order BGR, CHW)
- output tensor: float32[1,1,960,640] — DB text probability map in [0,1]
                 (Sigmoid baked in; DB box post-processing is NOT baked in — done in Dart)
- classes / labels: n/a (text/no-text probability map)

Then: trigger benchmark, wait for CONVERTING -> OPTIMIZING -> READY.

---

## Model 2 of 2 — RECOGNIZER (reads text from each cropped line)

Drag these into the dashboard:
- model:  en_ppocrv5_mobile_rec.onnx
- sample: en_ppocrv5_mobile_rec_sample_input.npy

Create the model with:
- name:    ajayshah/ReceiptScannerRec
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  float32[1,3,48,320], NCHW
                 (preprocess: resize line-crop to h=48 keep-aspect, right-pad to W=320,
                  /255 then (x-0.5)/0.5 -> [-1,1], channel order BGR, CHW)
- output tensor: float32[1,40,438] — CTC posteriors (softmaxed), 40 timesteps x 438 classes
                 (greedy CTC decode is NOT baked in — done in Dart)
- classes / labels: 438-class CTC head. blank = index 0, dict chars = 1..436,
                    space = index 437. Decode table shipped as rec_charset.json.

Then: trigger benchmark, wait for CONVERTING -> OPTIMIZING -> READY.

---

## Paste back to the agent (it is BLOCKED at GATE 0 until you do)

For BOTH models:
- the model name + version you registered (ReceiptScannerDet, ReceiptScannerRec)
- the served input/output shapes the dashboard shows, for each model
- modelMode: default RUN_AUTO
  (Do NOT use RUN_ACCURACY as a crash workaround - it isn't one. The iOS/macOS 26.3+
   CoreML-GPU crash is handled server-side by ZETIC filtering the GPU candidate; no client
   mode avoids it. See CLAUDE.md section 5.)
