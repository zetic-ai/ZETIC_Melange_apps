# Melange upload — PronunciationScoring

Drag these into the dashboard:
- model:  citrinet256_phoneme.onnx
- sample: sample_input.npy

Create the model with:
- name:    ajayshah/PronunciationScoring
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  float32[1, 81760] — raw mono 16 kHz waveform, 5.11 s, values in [-1, 1]
- output tensor: float32[1, 64, 45] — 64 CTC frames (80 ms hop) x 45 classes of
  log-softmax scores; classes = 39 ARPABET phonemes (ids 0-38) + 5 unused
  tokenizer specials (39-43) + CTC blank (44); see labels.txt
- classes / labels: labels.txt (45 entries)

Then: trigger benchmark, wait for CONVERTING -> OPTIMIZING -> READY.

Paste back to the agent (it is BLOCKED until you do):
- the model name + version you registered
- the served input/output shapes the dashboard shows
- modelMode: default RUN_AUTO
  (Do NOT use RUN_ACCURACY as a crash workaround — it isn't one. The iOS/macOS
  26.3+ CoreML-GPU crash is handled server-side by ZETIC filtering the GPU
  candidate; no client mode avoids it. See CLAUDE.md section 5.)
