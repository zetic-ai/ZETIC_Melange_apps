# Melange upload — SignTranslate

This app is a **TWO-MODEL pipeline**. Upload BOTH models as **separate** Melange models.
(The optional translate step is downstream Dart — there is NO third model to upload.)

---

## Model 1 of 2 — text DETECTOR

Drag these into the dashboard:
- model:  ppocrv5_mobile_det.onnx
- sample: ppocrv5_mobile_det_sample_input.npy

Create the model with:
- name:    ajayshah/SceneTextDetector
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  float32[1,3,736,736], NCHW
                 (BGR channel order; /255 then mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225])
- output tensor: float32[1,1,736,736] — single-channel text-probability map in [0,1]
                 (Sigmoid baked in; DBPostProcess / box extraction NOT baked — done in Dart)
- classes / labels: n/a (dense probability map, not a classifier)

Then: trigger benchmark, wait for CONVERTING -> OPTIMIZING -> READY.

---

## Model 2 of 2 — text RECOGNIZER

Drag these into the dashboard:
- model:  latin_ppocrv5_mobile_rec.onnx
- sample: latin_ppocrv5_mobile_rec_sample_input.npy

Create the model with:
- name:    ajayshah/SceneTextRecognizer
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  float32[1,3,48,320], NCHW
                 (BGR channel order; normalized (pixel/255 - 0.5)/0.5 -> range [-1,1];
                  each detected crop resized to height 48, width <=320, right-padded to 320)
- output tensor: float32[1,40,838] — 40 CTC time-steps x 838 classes, Softmax baked
                 (CTC greedy decode NOT baked — done in Dart)
- classes / labels: 838 CTC classes = blank@0 + 836 chars (latin_charset.txt) + space@837

Then: trigger benchmark, wait for CONVERTING -> OPTIMIZING -> READY.

---

## Paste back to the agent (it is BLOCKED at GATE 0 until you do) — for BOTH models

- the model name + version you registered (for EACH: SceneTextDetector, SceneTextRecognizer)
- the served input/output shapes the dashboard shows (for EACH model)
- modelMode: default RUN_AUTO
  (Do NOT use RUN_ACCURACY as a crash workaround - it isn't one. The iOS/macOS
  26.3+ CoreML-GPU crash is handled server-side by ZETIC filtering the GPU
  candidate; no client mode avoids it. See CLAUDE.md section 5.)
