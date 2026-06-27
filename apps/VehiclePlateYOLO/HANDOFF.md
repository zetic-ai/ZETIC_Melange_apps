Goal
A real-time, fully on-device license-plate detection demo for Flutter (iOS +
Android), powered by a single-pass YOLOv8n detector through the ZETIC Melange
SDK. Streams the live camera feed, runs detection each frame on-device inside a
long-lived dedicated inference isolate, and overlays labeled plate boxes with a
live plate-count + pipeline-latency HUD. Single class (license_plate), no OCR,
no two-stage vehicle-crop chain.

Todo List
[x] Create core Flutter structure (loading screen, camera screen, theme, HUD,
    detection overlay).
[x] Long-lived DEDICATED inference isolate: model create -> warm-up -> run ->
    close all owned inside the isolate (FFI handle is isolate-bound). Main
    isolate sends frame bytes in (one copy), gets List<Detection> back. _busy
    frame-guard drops frames rather than queueing.
[x] Melange lifecycle wrapper (ZeticMLangeModel.create -> Tensor.float32View ->
    run -> close), modelMode RUN_AUTO, version 1.
[x] personalKey via String.fromEnvironment('ZETIC_KEY'); never hardcoded/committed.
    Clear on-screen error if the define is empty.
[x] Preprocessing: letterbox 640x640 (pad 0.5), BGRA (iOS) / YUV420 (Android)
    -> RGB, /255 normalize, NCHW float32 [1,3,640,640], fused single reverse-
    mapped pass into a pre-allocated buffer.
[x] Post-processing: decode [1,5,8400] CHANNEL-major (stride c*8400+a),
    threshold-before-geometry (strict > 0.25), NO re-applied sigmoid (baked in),
    cxcywh->xyxy, un-letterbox, single-class GLOBAL NMS (IoU 0.45).
[x] Detection overlay (BoxFit.cover mapping, repaint-on-change) + HUD with live
    count, pipeline latency, and a buf/rot/img diagnostic line for device
    orientation confirmation.
[x] Tier A unit tests (9 named tests, 10 cases) + A4 hot-path micro-benchmark.
[x] Tier A1 analyze: `flutter analyze` -> No issues found (0 errors/0 warnings).
[x] Tier A3 unit tests: 10/10 pass.
[x] Tier A4 benchmark: median recorded (baseline 3.60ms -> 2.94ms after Tier B).
[x] Tier B: applied + measured one structural optimization (>0.5% rule);
    others applied by design (see Optimization log below).
[ ] [BLOCKED - human/macOS TCC] Commit to branch app/vehicleplate, apply the
    Tier B preprocessor optimization into the worktree, and write this
    HANDOFF.md into the worktree. Root cause: macOS revoked the terminal's Full
    Disk Access mid-session, so every read/write/git under ~/Desktop returns
    "Operation not permitted" (CLAUDE.md section 5). Fix: System Settings ->
    Privacy & Security -> Full Disk Access -> enable the terminal app, fully
    relaunch it (or move the repo off ~/Desktop). Then copy the validated files
    from the scratchpad mirror into the worktree and commit (steps below).
[ ] [BLOCKED - human] iOS signing/deploy config: team WVJ22PPYBP, iOS 16.6 min,
    NSCameraUsageDescription, vendored ZeticMLange.xcframework (via pod). Could
    not be written/run (TCC block + device-only). Mirror PyroGuard's setup.
[ ] [BLOCKED - human] Android release config: minSdk 24, AGP 8.9.1 / Kotlin
    2.1.0 / Gradle 8.11.1 pin, isMinifyEnabled=false, useLegacyPackaging.
[ ] Tier A2 release device build + physical-device run (device-only; needs the
    above signing config). Not yet run.
[ ] Confirm served runtimeApType on device console (expected NPU ~1.33ms; treat
    the console value as truth, not the dashboard number).

Deliverables
- Flutter source under apps/VehiclePlateYOLO/Flutter/ (screens, MelangeService
  with the dedicated isolate, preprocessor, postprocessor, NMS, detection model,
  overlay/HUD). NOTE: on disk in the worktree but UNCOMMITTED; the Tier-B-
  optimized preprocessor lives only in the scratchpad mirror until TCC is
  restored (see finish steps).
- Tier A tests under Flutter/test/ (9 files, 10 cases) + test/benchmark/
  hot_path_benchmark.dart.
- Model assets (already present, GATE 0): export.py, koushim-yolov8-license-
  plate.onnx, sample_input.npy, melange_upload.md, model_selection.md;
  registered Melange model ajayshah/VehiclePlateYOLO v1 (READY).

References
- App directory: apps/VehiclePlateYOLO
- Core SDK: ZETIC Melange (zetic_mlange 1.8.1, Flutter FFI; vendored device-only
  ZeticMLange.xcframework). SDK surface verified against the installed package.
- Model: YOLOv8n license-plate (Koushim/yolov8-license-plate-detection, MIT)
  input float32[1,3,640,640], output float32[1,5,8400] channel-major
  (cx,cy,w,h,plate_conf), single class license_plate, sigmoid baked in.
- Frameworks: Flutter 3.44.3, camera plugin, CoreML/ANE (iOS) & QNN/Hexagon
  (Android) via Melange.
- Test device (expected): physical iPhone (iOS 16.6+); Android minSdk 24.
