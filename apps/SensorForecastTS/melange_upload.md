# Melange upload — SensorForecastTS

Drag these into the dashboard:
- model:  chronos-bolt-tiny-ctx512.onnx
- sample: sample_input.npy

Create the model with:
- name:    ajayshah/SensorForecastTS
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  float32[1, 512]  — "context": raw (unnormalized) sensor values,
  a full sliding window of the 512 most recent samples (no NaN padding)
- output tensor: float32[1, 9, 64] — "quantile_preds": 9 quantiles (0.1, 0.2, …, 0.9;
  index 4 = median) × 64 future steps, already in original data units
- classes / labels: none (regression model — quantile forecast)

Then: trigger benchmark, wait for CONVERTING -> OPTIMIZING -> READY.

Paste back to the agent (it is BLOCKED until you do):
- the model name + version you registered
- the served input/output shapes the dashboard shows
- modelMode: default RUN_AUTO
  (Do NOT use RUN_ACCURACY as a crash workaround — it isn't one. The iOS/macOS
  26.3+ CoreML-GPU crash is handled server-side by ZETIC filtering the GPU
  candidate; no client mode avoids it. See CLAUDE.md section 5.)

Note: this is the same architecture + context length ZETIC runs in their own
Chronos demo (Team_ZETIC/Chronos-balt-tiny), so conversion is expected to be
uneventful. If it is not, compare against that dashboard project first.
