# Melange upload — VoxScribe (diarization half)

Drag these into the dashboard:
- model:  `pyannote_segmentation_static.onnx`   (5.98 MB, FP32, opset 13)
- sample: `sample_input.npy`                     (float32, shape [1, 1, 160000])

Create the model with:
- name:    `ajayshah/PyannoteSegmentation`
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  `x`  — float32[1, 1, 160000]  (batch=1, channels=1 mono, 160000 samples = 10.0 s @ 16 kHz)
- output tensor: `y`  — float32[1, 589, 7]      (batch=1, 589 frames, 7 powerset classes)
- value range:   raw waveform, roughly [-1, 1] float (NO mean/var normalization at the app boundary)
- post-processing baked in? NO. Powerset decode + window stitching + segment
  extraction are pure-Dart (see spec_stub_diarization.md).
- classes / labels: 7 **powerset** classes over a max of 3 local speakers, max 2
  simultaneous. Index 0 = "no speaker / silence". (Decode table in the spec.)

Then: trigger benchmark, wait for CONVERTING -> OPTIMIZING -> READY.

Notes for the upload (read before you click):
- This ONNX is already STATIC (no dynamic axes) and was constant-folded so it
  contains no Shape/Slice/If ops — it should sail through CONVERTING. Ops are all
  standard: Conv, InstanceNormalization, MaxPool, LSTM, MatMul, LogSoftmax, etc.
- The one op to watch on the device console is **LSTM** (pyannote's 2x BiLSTM).
  If a backend can't take LSTM it falls back to CPU — that is fine for the demo
  (one 10 s window is cheap). Record the served `runtimeApType` regardless.
- Do NOT upload `model.int8.onnx`; we intentionally ship FP32 and let Melange
  pick precision.

Paste back to the agent (it is BLOCKED at GATE 0 until you do):
- the model name + version you registered (expected `ajayshah/PyannoteSegmentation` v1)
- the served input/output shapes the dashboard shows (confirm [1,1,160000] -> [1,589,7])
- modelMode: default **RUN_AUTO**
  (Do NOT use RUN_ACCURACY as a crash workaround — it isn't one. The iOS/macOS
  26.3+ CoreML-GPU crash is handled server-side by ZETIC filtering the GPU
  candidate; no client mode avoids it. See CLAUDE.md section 5.)
- the served `runtimeApType` (NPU / GPU / CPU) from the device console, if known.
