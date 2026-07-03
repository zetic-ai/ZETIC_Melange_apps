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
[x] Secrets wiring: gitignored lib/config/secrets.dart (const zeticPersonalKey)
    + committed secrets.example.dart placeholder; git check-ignore verified
    BEFORE first commit (rule at Flutter/.gitignore). Key never in repo/logs.
[x] Verified installed zetic_mlange 1.8.1 API surface from pub-cache sources:
    create(personalKey:, name:, version:, modelMode:, onProgress:), sync
    run(List<Tensor>), Tensor.float32List, asFloat32List (VIEW over reused
    native buffer — copied), close(), isClosed.
[x] Core Flutter structure: loading screen (download progress + warm-up +
    retry-on-error), main screen (chart/HUD/events/controls), dark theme.
[x] Data feed service (GATE-2 ruling applied: SYNTHETIC ONLY, no NAB asset):
    seeded deterministic generator with two modes — "Machine replay"
    (industrial temperature arc with scripted failure segment per 6000-sample
    loop) and "Lab signal" (two-tone sine) — plus spike / level-shift /
    noise-burst injection buttons; 20 samples/s clock; pre-seeds the full
    512-sample window so forecasting starts on the first tick.
[x] [RESOLVED – GATE-2 ruling] NAB bundling: DENIED (AGPL-3.0). NAB stays a
    local-only Stage-0 validation artifact; the app ships the license-clean
    app-generated replay instead. No third-party data files in repo or assets.
[x] Melange lifecycle wrapper (create -> warmUp dummy run -> run -> close),
    _busy drop-not-queue guard, native-run latency captured for the HUD.
[x] Preprocessor: pre-allocated 512-slot ring buffer (SampleWindow) ->
    oldest-to-newest Float32List [1,512], raw values, no normalization;
    snapshotInto THROWS on a partial window (full-window contract enforced).
[x] Postprocessor: quantile-major decode (flat q*64+t, copies the SDK view);
    anomaly score max(0,(x-q90)/iqr,(q10-x)/iqr), iqr floor 1e-6; threshold
    slider 0.5-3.0 (default 1.0), 2-consecutive debounce; re-forecast every 8
    samples via pure ForecastPipeline (SDK-free, fully unit-tested).
[x] UI: scrolling live chart (CustomPainter, repaint keyed on a revision
    counter) with q10-q90 + q30-q70 fan and median line re-anchoring per
    forecast, "now" divider, anomaly rings + event list, HUD (infer ms, dart
    ms, score/threshold, event count, model line), injection buttons.
[x] Tier A3: 36 tests green across ring_buffer / quantile_decode /
    anomaly_score / threshold_debounce / forecast_alignment /
    no_normalization / data_feed suites (hand-built tensors, known outputs).
[x] Tier A4: hot-path micro-benchmark (test/benchmark/hot_path_benchmark.dart)
    — per-tick median 37 ns, per-forecast median 306 ns, total pure-Dart
    budget ~1.5 us per second of demo at 20 sps + 2.5 forecasts/s.
[x] Tier B: pass complete. Measured: setRange bulk snapshot 46 ns vs naive
    per-element loop 2809 ns (61x, kept). Other levers pre-applied by design
    or skipped with justification — see GATE-3 Tier B log (budget is ~1.5
    us/s; nothing else can clear the 0.5% rule meaningfully).
[x] Launcher icon: 1024x1024 domain glyph (teal trace, red anomaly ring, blue
    forecast fan, "now" divider) generated via flutter_launcher_icons for iOS
    (alpha removed) + Android.
[x] Product name "SentryWave": CFBundleDisplayName, android:label,
    MaterialApp(title:), app-bar + loading-screen title set. Bundle id
    com.zeticai.sensorforecastts, folder, and Melange name unchanged.
[x] A1 flutter analyze: zero errors, zero warnings, zero infos.
[x] iOS release build compiles (flutter build ios --release --no-codesign;
    signing itself is the human's device step) with iOS 16.6 target; Android
    minSdk 24 configured and release APK build verified.
[ ] [BLOCKED – human, GATE 3] Physical-device run: signing (team WVJ22PPYBP
    per PyroGuard), Developer Mode, first-launch model download on real
    network, served-artifact readout from the native console, multi-cold-start
    acceptance. Tier C checklist delivered at GATE 3.

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
