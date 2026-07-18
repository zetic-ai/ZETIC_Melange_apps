# Melange upload — LiveGate (TWO models)

This app is a **two-model on-device KYC gate**: a liveness/anti-spoof classifier (PAD)
plus a face-recognition embedding for 1:1 match. Upload BOTH as **separate** Melange
models. Do the two uploads independently; paste back the registered names/versions +
served shapes for **both** (the worker build runs in parallel — this only unblocks the
name/version injection and the device run).

Both ONNX files are already **fully STATIC** (batch pinned to 1, no dynamic axes, no
`Shape`/`If`/`Loop`/`Gather` ops — verified programmatically in `export.py`) and FP32
(Melange owns precision — no fp16 baked in), so they should sail through CONVERTING.
Ops are all standard (Conv/BatchNorm/PRelu/Gemm/MatMul/Reshape).

---

## Model 1 of 2 — PAD / ANTI-SPOOF (liveness)

Drag these into the dashboard:
- model:  `pad_minifasnet_v2.onnx`     (1.76 MB, FP32, opset 12 — MiniFASNet-V2 2.7_80x80)
- sample: `sample_input_pad.npy`       (float32, shape [1, 3, 80, 80])

Create the model with:
- name:    `ajayshah/LiveGatePAD`
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  `input` — float32[1, 3, 80, 80], NCHW, **BGR, range [0,255] (NOT /255)**,
                 fed a 2.7× face-margin crop resized to 80×80
- output tensor: `output` — float32[1, 3] raw logits → softmax; **LIVE = softmax[class 1]**
                 (class 0 & 2 are the two spoof types)
- post-processing baked in? **NO.** softmax + threshold is pure-Dart.
- classes / labels: `[spoof_a, LIVE, spoof_b]` — index 1 = live (empirically verified;
  the source model card's ordering/normalization are BOTH wrong — see model_selection.md).

Then: trigger benchmark, wait for CONVERTING → OPTIMIZING → READY.

---

## Model 2 of 2 — FACE EMBEDDING (1:1 match)

Drag these into the dashboard:
- model:  `face_sface.onnx`            (38.55 MB, FP32, opset 12 — SFace / MobileFaceNet ArcFace)
- sample: `sample_input_face.npy`      (float32, shape [1, 3, 112, 112])

Create the model with:
- name:    `ajayshah/LiveGateFace`
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  `data` — float32[1, 3, 112, 112], NCHW, **BGR, range [0,255]** (the
                 (x-127.5)/128 normalization is baked into the graph), fed a 112×112
                 aligned face crop
- output tensor: `fc1` — float32[1, 128] face embedding, **NOT L2-normalized in-graph**
- post-processing baked in? **NO.** L2-normalize in Dart, then cosine vs the enrolled
  reference vector; MATCH if cosine ≥ ~0.363.
- classes / labels: N/A — this is a 128-d embedding, not a classifier.

Then: trigger benchmark, wait for CONVERTING → OPTIMIZING → READY.

---

## Paste back to the agent (unblocks name/version injection + device run — for BOTH models)

For **LiveGatePAD**:
- registered model name + version (expected `ajayshah/LiveGatePAD` v1)
- served input/output shapes the dashboard shows (confirm [1,3,80,80] → [1,3])
- served `runtimeApType` (NPU / GPU / CPU) from the device console, if known

For **LiveGateFace**:
- registered model name + version (expected `ajayshah/LiveGateFace` v1)
- served input/output shapes the dashboard shows (confirm [1,3,112,112] → [1,128])
- served `runtimeApType` (NPU / GPU / CPU) from the device console, if known

modelMode for both: default **RUN_AUTO**.
- Do **NOT** use RUN_ACCURACY as a crash workaround — it is not one. The iOS/macOS 26.3+
  CoreML-GPU MPSGraph crash is handled **server-side** by ZETIC filtering the GPU candidate;
  no client modelMode avoids it (all four returned the same crashing artifact on PyroGuard).
  See CLAUDE.md section 5.
- "Benchmarked" ≠ "served": a fast NPU row in the report may never be served for a given
  chip. Read the *served* target+apType from the native console — that is ground truth.

Note: the dashboard header shows `ZETIC | <Name>` — that `ZETIC |` is the org/workspace
DISPLAY prefix, NOT the account. The SDK name is `ajayshah/<Name>` WITH the slash. The
dashboard does NOT echo a version — first upload = version 1, confirmed at first SDK `create()`.
The registered name may legitimately differ from the proposal — that is a cheap one-constant
rename, not an error.
