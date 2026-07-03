Goal
A fully on-device sensor forecasting + anomaly detection demo for Flutter (iOS
first), powered by amazon/chronos-bolt-tiny through the ZETIC Melange SDK
(ajayshah/SensorForecastTS v1, RUN_AUTO). Streams a replayed/synthetic sensor
feed, runs a 512-sample sliding-window inference, draws a live chart with a
64-step quantile forecast fan (q10–q90 band + median), and flags readings that
break out of the predicted band (debounced band-exceedance score), with a
latency + score HUD. Product name: "SentryWave" (display name only; bundle id,
folder, and Melange model name stay SensorForecastTS).

Todo List
[x] Stage 0: model search, top-5 shortlist, winner rationale
    (model_selection.md — amazon/chronos-bolt-tiny, Apache-2.0).
[x] Stage 0: static ONNX export recipe (export.py — [1,512]->[1,9,64], opset 12,
    zero dynamic dims, max |onnx-torch| 1.1e-05) + sample_input.npy.
[x] Stage 0: behavioral validation on NAB machine-temperature (4/4 labeled
    failure windows @ thr 1.0, 1.07% raw FP) + synthetic injected anomalies
    (score separation 4.5–14.3 vs clean p99 0.40).
[x] GATE 0: Melange registration by human — ajayshah/SensorForecastTS v1,
    READY; served shapes echo the export exactly; benchmark 100% deployable
    (NPU med 0.94 ms / CPU med 5.77 ms).
[ ] Secrets wiring: gitignored lib/config/secrets.dart (const zeticPersonalKey)
    + committed secrets.example.dart placeholder; VERIFY git check-ignore
    catches the real file before any commit. Key itself never in repo/logs.
[ ] Verify installed zetic_mlange package version and its exact API surface
    (create(personalKey:, name:), run(List<Tensor>), Tensor.float32List,
    asFloat32List, close) before coding against it.
[ ] Core Flutter structure: loading screen (model download + warm-up progress),
    main screen (live chart + HUD), theme.
[ ] Data feed service: bundled-CSV replay (pre-seeds the 512-sample window so
    inference starts instantly) + seeded synthetic generator (two-tone sine +
    noise) with spike / level-shift / noise-burst injection buttons;
    configurable replay speed (default ~20 samples/s).
[ ] [BLOCKED – orchestrator, GATE-2 question] Bundled real-data replay: NAB
    corpus is AGPL-3.0 — confirm bundling machine_temperature_system_failure.csv
    as an asset is acceptable for a GTM demo, else ship the app-generated
    realistic replay track instead (functional fallback already planned).
[ ] Melange lifecycle wrapper (create -> warm-up dummy run -> run -> close),
    _busy guard, latency capture per stage.
[ ] Preprocessor: pre-allocated 512-slot ring buffer -> oldest-to-newest
    Float32List [1,512], raw values, no normalization, full-window contract.
[ ] Postprocessor: quantile-major decode (flat q*64+t) -> fan series; anomaly
    score = max(0,(x-q90)/iqr,(q10-x)/iqr), iqr floor 1e-6; threshold 1.0 with
    2-consecutive debounce; re-forecast every 8 samples.
[ ] UI: scrolling live chart (CustomPainter) with forecast fan re-anchoring on
    each re-forecast, anomaly markers + event list (timestamp, score), HUD
    (inference ms, score, served-artifact line), injection buttons.
[ ] Tier A: unit tests per approved list (windowing, decode layout, score math,
    threshold/debounce boundaries, alignment, no-normalization, full-window).
[ ] Tier A: hot-path micro-benchmark (test/benchmark/) — decode + score +
    debounce on mock [1,9,64] tensors, median over many iterations.
[ ] Tier B: optimization pass with measured before/after deltas (0.5% rule).
[ ] Launcher icon: 1024x1024 domain glyph (waveform + sentry/shield motif),
    flutter_launcher_icons, remove_alpha_ios: true, iOS + Android.
[ ] Product name "SentryWave": CFBundleDisplayName, android:label,
    MaterialApp(title:) / app-bar + loading text.
[ ] iOS signing/deploy config (team, min iOS 16.6), release-mode device build;
    Android minSdk 24 config (build verified; device run is human-owned).
[ ] Tier C runtime-risk checklist for GATE 3 (served-artifact expectation,
    devicectl console command, cold-start/network note, N-runs acceptance).

Deliverables
- Flutter source under apps/SensorForecastTS/Flutter/ (screens, services:
  melange_service / data_feed / preprocessor / postprocessor, models, widgets:
  live chart + HUD + event list).
- Model assets: export.py, chronos-bolt-tiny-ctx512.onnx (local, gitignored per
  repo policy), sample_input.npy (local, gitignored), registered Melange model
  ajayshah/SensorForecastTS v1.
- Tests: Tier A suite + hot-path micro-benchmark with recorded medians.
- Docs: SPEC.md (final), model_selection.md, melange_upload.md, this
  HANDOFF.md kept living through the build.

References
- App directory: apps/SensorForecastTS (branch explore/sensor-ts, worktree
  /Users/ajayshah/Desktop/ZETIC/explore-sensor-ts-wt)
- Core SDK: ZETIC Melange (zetic_mlange Flutter plugin — version to be pinned
  after install check)
- Model: amazon/chronos-bolt-tiny via Melange ajayshah/SensorForecastTS v1
  (input context float32[1,512] raw values; output quantile_preds
  float32[1,9,64], quantiles 0.1–0.9, index 4 = median, original units)
- Validation data: NAB machine_temperature_system_failure (Stage-0 ground
  truth; in-app bundling pending license decision) + seeded synthetic signals
- Prior art: apps/FireDetectionYOLO (PyroGuard — SDK realities, HUD-based
  observability, release-build workflow), apps/ChronosTimeSeries (ZETIC's own
  Chronos-Bolt demo, native iOS/Android)
- Test device: human's iPhone (per GATE-3 run; PyroGuard used iPhone 15, iOS 26.5)
