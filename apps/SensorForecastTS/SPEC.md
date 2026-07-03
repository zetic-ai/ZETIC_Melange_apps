# SPEC: SensorForecastTS  (FINAL — GATE 0 cleared 2026-07-02, dashboard READY)

## One-line pitch
Live on-device sensor forecasting with uncertainty bands and real-time anomaly flags —
predictive-maintenance demo for industrial prospects (streams/replays a sensor feed,
draws the 64-step forecast fan, and flags readings that break out of the predicted band).

## Model
- Source (HF repo / origin): amazon/chronos-bolt-tiny (Apache-2.0)
- Architecture: Chronos-Bolt (T5-style encoder-decoder, single-pass direct multi-step
  quantile head — NOT autoregressive; one inference = full forecast)
- Melange model name: ajayshah/SensorForecastTS (SDK `create(name:)` value WITH the
  slash. The dashboard header shows "ZETIC | SensorForecastTS" — "ZETIC |" is a display
  prefix only, never part of the name.)
- Melange version: 1
- Served input/output shapes (dashboard echo, status READY): context float32[1,512];
  quantile_preds float32[1,9,64] — exactly as exported.
- Dashboard benchmark (recorded for context; binding caveat CLAUDE.md §5 "benchmarked ≠
  served"): 100% deployable, FP32 across Apple/Samsung/Other, 3 quantizations, model
  size 8.14–32.51 MB; latency across devices NPU min 0.30 / med 0.94 / avg 2.05 ms,
  GPU med 8.94 ms, CPU med 5.77 ms; accuracy 13.87–54.79 dB SNR; memory load ≤125 MB,
  inference 11.14–125.51 MB. Even a CPU-served artifact (~6 ms) is demo-fine here.
- Input tensor: float32[1, 512], layout [batch, time]; RAW sensor values in original
  units — NO client-side normalization (instance norm + de-norm are inside the graph).
  The window MUST be completely filled with real samples (this export does not support
  NaN padding — the observed-mask is baked to all-ones).
- Output tensor: float32[1, 9, 64] = [batch, quantile, horizon]. Quantile levels, in
  index order 0..8: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 (index 4 = median).
  Values are already de-normalized to original data units. Layout is quantile-major:
  flat index = q * 64 + t.
- Post-processing baked into ONNX? Yes for de-normalization; nothing else needed
  beyond indexing quantiles. No NMS/activation concerns.
- Classes / labels: none (regression).
- modelMode to use and why: RUN_AUTO (registered default). No client mode can steer
  backend selection anyway (PyroGuard lesson); treat the served target+apType from the
  native console as ground truth, not the requested mode.

## Input source
- No camera/mic. Data feed is (a) bundled CSV replay of a real industrial sensor
  (recommended: NAB machine_temperature_system_failure.csv, 5-min cadence, 4 real
  labeled failure events — validated in Stage 0; LICENSE FLAG raised at GATE 2: the NAB
  corpus is AGPL-3.0, so bundling the CSV as an app asset needs an explicit orchestrator
  OK — fallback is an app-generated realistic replay series) played back at demo speed
  (e.g. 10–30 samples/s), and/or (b) a synthetic signal generator (two-tone sine +
  noise) with user-triggerable injected anomalies (spike / level-shift / noise-burst
  buttons — the crowd-pleaser at a booth).
- Sample rate requested: n/a (replay clock is app-defined).
- Orientation handling: n/a.

## Pre-processing pipeline (ordered, exact)
1. Maintain a ring buffer of the most recent 512 float sensor samples (Float32List,
   pre-allocated once).
2. Do NOT normalize, detrend, or scale — feed raw values (in-graph instance norm).
3. Wait until the buffer holds 512 REAL samples before the first inference (pre-seed
   from the replay file's history so the demo starts instantly; never pad with NaN or
   repeats).
4. On each inference tick (e.g. every N new samples), copy buffer in time order
   (oldest → newest) into shape [1, 512] and wrap as Tensor.float32List.

## Post-processing pipeline (ordered, exact)
1. Read output as Float32List of length 576; reshape as [9, 64], quantile-major
   (flat index = q * 64 + t).
2. Forecast fan: median = row 4; band = rows 0 (q10) and 8 (q90); optionally rows
   2/6 (q30/q70) for a two-tone fan. Horizon step t corresponds to sample time
   now + (t+1) ticks.
3. Anomaly score for each incoming actual sample x that falls at horizon step t of the
   most recent forecast:
   iqr   = max(q90[t] - q10[t], 1e-6)
   score = max(0, (x - q90[t]) / iqr, (q10[t] - x) / iqr)
4. Flag anomaly when score >= 1.0 for >= 2 consecutive samples (debounce — Stage-0
   measured: threshold 1.0 gives 4/4 NAB failure windows at 1.07% raw FP rate;
   the 2-consecutive rule suppresses most of that 1%). Expose the threshold as a
   settings slider (0.5–3.0) if desired.
5. Re-forecast cadence: every 8–12 new samples (Stage-0 validation used 12) or on
   demand; scores between re-forecasts use the latest forecast's later horizon steps.

## UI
- Left to the worker. Functional must-haves: live scrolling chart of recent history;
  forecast fan (median line + shaded q10–q90 band) extending ahead of "now"; anomaly
  flags rendered on the trace (and a persistent event list with timestamp + score);
  live anomaly-score readout; inference latency readout; anomaly-injection buttons in
  synthetic mode.

## Platform targets
- iOS 16.6+, Android minSdk 24 (match PyroGuard).
- Known OS traps: standard T5 attention (softmax/matmul) — same family of graphs as the
  ViT/YOLO heads implicated in the iOS 26.3+ MPSGraph FP32-GPU crash; ZETIC filters GPU
  server-side for affected OS versions (NOT a selection criterion, not client-fixable).
  Read the served target+apType from the native console; budget for CPU-speed fallback
  (irrelevant here: model is ~1 ms-class on CPU, so even CPU serving is demo-fine).

## Validation focus (Tier A traps specific to THIS model)
- Ring-buffer windowing: hand-built sequence → assert the tensor is oldest→newest with
  exactly the last 512 samples (off-by-one and reversal traps).
- Quantile-major decode: hand-built 576-float tensor with one known value per (q,t) →
  assert flat index q*64+t (NOT t*9+q).
- Band/score math: known q10/q50/q90 and known x → exact expected score, including
  the max(0, ·) clamp and iqr floor.
- Threshold boundary: score just-below 1.0 not flagged; just-above flagged only after
  2 consecutive.
- Forecast/time alignment on the chart: horizon step t plots at now+(t+1), and the fan
  re-anchors correctly after each re-forecast.
- NO-normalization contract: assert preprocessing performs no scaling (feed values in
  the hundreds; forecast must come back in the same units).
- Full-window contract: inference is never invoked with <512 real samples.

## Stage-0 measured baseline (for honesty in the demo script)
- Synthetic periodic signal: median MAE ~9× better than last-value naive; coverage 0.83.
- NAB machine temperature: 4/4 labeled failure windows detected @ threshold 1.0,
  1.07% raw FP; 64-step point forecast ≈ persistence on this near-random-walk series
  (the calibrated band, not point accuracy, is the demo story there).
- Degeneracy checks passed (constant/Gaussian/tiny-scale inputs, monotonic quantiles).
- Desktop CPU latency ~1.2 ms (onnxruntime); expect SDK overhead to dominate on-device.
