# Melange upload — LiveDocRedact (TWO models)

This app is a **two-model OCR pipeline**. Upload BOTH models as **separate** Melange
models. Do the two uploads independently; the app is BLOCKED at GATE 0 until you paste
back the registered names/versions + served shapes for **both**.

Both ONNX files are already **fully STATIC** (no dynamic axes, no `Shape`/`If`/`Slice`
ops — verified programmatically in `export.py`) and FP32 (Melange owns precision — no
fp16 baked in), so they should sail through CONVERTING. Ops are all standard
(Conv/BatchNorm/ConvTranspose/Sigmoid/Resize for the detector; Conv/MatMul/Softmax/CTC
for the recognizer).

---

## Model 1 of 2 — TEXT DETECTOR

Drag these into the dashboard:
- model:  `doc_text_detector.onnx`        (4.75 MB, FP32, opset 12 — DBNet)
- sample: `detector_sample_input.npy`     (float32, shape [1, 3, 640, 640])

Create the model with:
- name:    `ajayshah/DocTextDetector`
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  `x` — float32[1, 3, 640, 640], NCHW (BGR, ImageNet-normalized
                 (pixel/255 − mean)/std; mean [0.485,0.456,0.406] std [0.229,0.224,0.225])
- output tensor: `fetch_name_0` — float32[1, 1, 640, 640] (single text-probability heatmap
                 in 640×640 space; ~0..1)
- post-processing baked in? **NO.** DB decode (binarize → contours → unclip → quad boxes)
  is pure-Dart.
- classes / labels: N/A — this is a per-pixel text/no-text heatmap, not a classifier.

Then: trigger benchmark, wait for CONVERTING → OPTIMIZING → READY.

---

## Model 2 of 2 — TEXT RECOGNIZER

Drag these into the dashboard:
- model:  `doc_text_recognizer.onnx`      (7.8 MB, FP32, opset 12 — CRNN/SVTR CTC)
- sample: `recognizer_sample_input.npy`   (float32, shape [1, 3, 48, 320])

Create the model with:
- name:    `ajayshah/DocTextRecognizer`
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  `x` — float32[1, 3, 48, 320], NCHW (BGR, PP-OCR rec norm
                 (pixel/255 − 0.5)/0.5 → [-1,1]; **fixed width 320**, aspect-resize+pad)
- output tensor: `fetch_name_0` — float32[1, 40, 438] (40 CTC steps × 438 classes, softmax)
- post-processing baked in? **NO.** Greedy CTC decode (argmax → collapse repeats → drop
  blank) is pure-Dart, using `en_dict.txt`.
- classes / labels: CTC label list = **[blank](0) + 436 chars from `en_dict.txt` (1..436) +
  space ' ' (437)**. (`en_dict.txt` ships in this folder for the Dart decoder.)

Then: trigger benchmark, wait for CONVERTING → OPTIMIZING → READY.

---

## Paste back to the agent (it is BLOCKED at GATE 0 until you do — for BOTH models)

For **DocTextDetector**:
- registered model name + version (expected `ajayshah/DocTextDetector` v1)
- served input/output shapes the dashboard shows (confirm [1,3,640,640] → [1,1,640,640])
- served `runtimeApType` (NPU / GPU / CPU) from the device console, if known

For **DocTextRecognizer**:
- registered model name + version (expected `ajayshah/DocTextRecognizer` v1)
- served input/output shapes the dashboard shows (confirm [1,3,48,320] → [1,40,438])
- served `runtimeApType` (NPU / GPU / CPU) from the device console, if known

modelMode for both: default **RUN_AUTO**.
- Do **NOT** use RUN_ACCURACY as a crash workaround — it is not one. The iOS/macOS 26.3+
  CoreML-GPU MPSGraph crash is handled **server-side** by ZETIC filtering the GPU candidate
  for the affected OS; no client modelMode avoids it (all four returned the same crashing
  artifact on PyroGuard). See CLAUDE.md section 5.
- "Benchmarked" ≠ "served": a fast NPU row in the report may never be served for a given
  chip. Read the *served* target+apType from the native console — that is ground truth.
