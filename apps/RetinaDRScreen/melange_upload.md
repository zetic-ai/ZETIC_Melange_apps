# Melange upload — RetinaDRScreen

Drag these into the dashboard:
- model:  mobilenetv2-dr-referable.onnx
- sample: sample_input.npy

Create the model with:
- name:    ajayshah/RetinaDRScreen
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  float32[1,3,224,224], NCHW, RGB.
                 Value range is NOT plain 0-1. The Dart preprocessor owns this
                 (see SPEC_STUB.md) and must reproduce the model's MobileNetV2
                 pipeline EXACTLY:
                   1. resize shortest edge -> 256 (bilinear), preserve aspect
                   2. center-crop 224 x 224
                   3. float32, * 1/255                  -> [0,1]
                   4. normalize (v - 0.5) / 0.5         -> [-1,1]
                      (mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
                   5. HWC -> NCHW [1,3,224,224], RGB channel order
- output tensor: float32[1,2], RAW LOGITS (2 values; NOT softmaxed).
                 Apply softmax downstream in Dart. P(referable) = softmax[index 1].
                 Decision: referable if P(index 1) >= threshold (default 0.5;
                 the app may expose the threshold).
- classes / labels (index -> label):
    0 = Nrdr  (NOT referable — no DR / mild, DR grade 0-1)
    1 = Rdr   (REFERABLE — DR grade >= 2, Moderate or worse)

Then: trigger benchmark, wait for CONVERTING -> OPTIMIZING -> READY.

Paste back to the agent (it is BLOCKED until you do):
- the model name + version you registered
- the served input/output shapes the dashboard shows
- modelMode: default RUN_AUTO
  (Do NOT use RUN_ACCURACY as a crash workaround - it isn't one. The iOS/macOS
  26.3+ CoreML-GPU crash is handled server-side by ZETIC filtering the GPU
  candidate; no client mode avoids it. See CLAUDE.md section 5.)

---

## ⚠️ LICENSE — PRE-SHIP LEGAL CHECK (do NOT skip)

The Hugging Face repo `EscvNcl/MobileNet-V2-Retinopathy` declares **`license: other`
with NO stated terms**. The base model `google/mobilenet_v2` is Apache-2.0, but the
fine-tuned DR weights' redistribution/commercial terms are **UNDECLARED**. Before this
app ships in ZETIC's GTM / trade-show distribution, get the license clarified (contact
the author or confirm the training-data terms). This is a legal gate, not a footnote —
an undeclared license can sink a commercial demo. See model_selection.md.
