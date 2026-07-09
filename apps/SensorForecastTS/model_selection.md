# Model selection — SensorForecastTS (time-series forecasting / anomaly detection, use-case: sensor forecast + anomaly flags for industrial predictive maintenance)

First exploration of the time-series family — this run also establishes the family's
export recipe (see `export.py` header).

## Search
`HfApi().list_models(filter="time-series-forecasting", sort="downloads")` plus free-text
sweeps (`chronos-bolt`, `patchtst`, `dlinear`, `tcn forecast`, `timesfm`, `moirai`,
`anomaly detection time series`, `lstm autoencoder anomaly`). 2026-07-02.

## Shortlist (top 5)
| Rank | HF repo | Downloads | License | Export path | Melange-fit notes | Score |
|------|---------|-----------|---------|-------------|-------------------|-------|
| 1 | amazon/chronos-bolt-tiny | 1.09M (+2.49M via autogluon mirror) | Apache-2.0 | torch trace → ONNX (this repo's recipe, verified this run) | 8.65M params → 34.9 MB fp32 ONNX. NOT autoregressive: one forward pass = full 64-step, 9-quantile forecast → single static graph, zero-shot on any sensor series, quantile band = free anomaly score. **ZETIC themselves already run this exact model on Melange** (Team_ZETIC/Chronos-balt-tiny dashboard project + `apps/ChronosTimeSeries`), so the compile path is proven. Static [1,512]→[1,9,64] confirmed, opset 12, no dynamic dims. | 9.5 |
| 2 | ibm-granite/granite-timeseries-ttm-r2 | 365K | Apache-2.0 | tsfm_public → torch → ONNX (standard) | TinyTimeMixer, ~1-5M params, pure MLP-mixer (no attention) — would export trivially and is the fallback if Melange ever chokes on T5 attention. BUT point forecasts only: no quantile band, so anomaly scoring needs a hand-rolled residual threshold; weaker demo story. Extra dep (granite-tsfm). | 8.0 |
| 3 | amazon/chronos-bolt-mini | 0.70M (autogluon) + 0.10M | Apache-2.0 | same recipe as winner | Same family, 21M params → ~84 MB fp32: past the mobile sweet spot for no visible demo gain over tiny. | 6.5 |
| 4 | AutonLab/MOMENT-1-small | 704K | MIT | torch → ONNX, but task-head config required | 37M-param T5 foundation model; supports anomaly detection via reconstruction, but needs per-task head wiring, ~150 MB fp32, heavier export with more dynamic machinery. Oversized for this demo. | 5.0 |
| 5 | Salesforce/moirai-2.0-R-small | 473K | **CC-BY-NC-4.0 — FAILS the commercial-demo gate** | uni2ts → ONNX (nontrivial) | Good model, unusable license for a GTM demo. Listed to record the loud license flag, per rubric. | 3.0 |

Also considered and rejected: google/timesfm-2.5-200m (Apache but 200M params — LLM-scale,
disqualified on size), NeoQuasar/Kronos-* (MIT but finance K-line-specific tokenizer +
autoregressive decode — poor task fit and poor static-export fit),
ibm-granite/granite-timeseries-patchtst (trained on ETTh1 only, not zero-shot — would
need per-dataset retraining), keras-io/time-series-anomaly-detection-autoencoder (CC0
toy, 6 downloads, single-series conv-AE — no quality signal), DLinear/TCN checkpoints
(essentially no maintained pretrained weights on HF; would mean training our own).

## Winner: amazon/chronos-bolt-tiny
- **Melange-fit is proven, not predicted**: ZETIC's own ChronosTimeSeries demo runs this
  exact architecture (context 512) on Melange, and our export reproduces a fully static
  [1,512]→[1,9,64] graph at opset 12 with zero dynamic dims and 1.1e-05 max drift vs torch.
- **Demo-fit**: zero-shot — works on ANY replayed/streamed sensor without retraining; the
  9-quantile output gives forecast + uncertainty band + a principled anomaly score
  (band exceedance) from ONE model call. Measured: 4/4 NAB labeled machine-failure
  windows detected at 1.07% FP (see Validation below).
- Over TTM-r2 (runner-up): quantiles beat point forecasts for the anomaly story, and the
  proven-on-Melange evidence outweighs TTM's simpler op set.
- 34.9 MB fp32 is at the top of the "low tens of MB" band but within it; Melange handles
  precision downstream.

## Export
- Recipe: `export.py` (THE time-series family recipe, first of its kind — reuse for any
  Chronos-Bolt size). Key moves: monkeypatch out dynamic control flow, patch
  `InstanceNorm` to drop unexportable `aten::nanmean`, hardcode observed-mask to ones
  (full-window contract, no NaN padding), replace `Patch.unfold` with a pure reshape
  (legacy exporter mis-decomposes unfold — missing transpose, verified).
- Input:  `context` float32[1, 512] — raw, UNNORMALIZED sensor values (instance
  normalization is inside the graph). Must be a FULL window of real values.
- Output: `quantile_preds` float32[1, 9, 64] — quantiles 0.1…0.9 (index 4 = median) ×
  64 future steps, already de-normalized to original data units. No further
  post-processing baked in or needed beyond quantile indexing.
- Opset 12 (13/14 also verified working; 12 chosen as the family's known-good baseline).
  Static shapes confirmed by assertion in `export.py` (checker + shape-inference pass,
  zero dynamic dims).

## Validation (measured, onnxruntime, exact app pipeline — honest numbers)
Data: NAB `machine_temperature_system_failure` (real industrial sensor, 5-min cadence,
4 labeled failure windows) + synthetic two-tone sine with injected spike / level-shift /
noise-burst anomalies.

**Forecast quality**
- Synthetic periodic signal: median-forecast MAE 0.63 vs last-value-naive 5.76 (~9×
  better); q10–q90 coverage 0.83 (nominal 0.80). The forecast visibly tracks the
  waveform — this is the live-demo behavior.
- NAB machine temperature, 87 rolling windows, 64-step horizon: MAE 4.17 vs
  persistence 4.14 (MASE≈1.01), vs daily-seasonal naive 10.54 (MASE 0.40). HONEST
  CAVEAT: at 5-min cadence this series is near random-walk, so NO forecaster
  meaningfully beats persistence at a 64-step horizon; the model is ~2.5× better than
  the seasonal baseline and its value here is the calibrated band, not point accuracy.
- Band calibration on NAB: 0.73 observed vs 0.80 nominal — slightly overconfident.

**Anomaly detection** (score = exceedance beyond [q10,q90], normalized by band width;
re-forecast hourly, threshold 1.0):
- NAB: **4/4 labeled failure windows detected**, false-positive rate 1.07% of normal
  points (normal-region p99 = 1.03, max 5.21 — a few normal-region excursions exist;
  the app should debounce, e.g. flag on ≥2 consecutive exceedances).
- Synthetic injected anomalies: spike → score 11.1, level-shift → 4.5, noise-burst →
  14.3, vs clean-region p99 = 0.40. Separation is unambiguous.

**Degeneracy checks**: Gaussian(100,5) → median ≈100; constant 42 → median 42.00 with
zero-width band; tiny-scale input (1e-3) handled (in-graph norm); quantiles monotonic
across all 9 levels. CPU latency (onnxruntime, M-series): ~1.2 ms — post-Melange
on-device latency will be dominated by SDK overhead, not compute.

Verdict: NOT degenerate, demo-strong. Ship it.
