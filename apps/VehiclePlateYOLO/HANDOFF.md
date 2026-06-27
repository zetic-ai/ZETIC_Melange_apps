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
    mapped pass into a pre-allocated buffer (inline-BGRA hot path).
[x] Post-processing: decode [1,5,8400] CHANNEL-major (stride c*8400+a),
    threshold-before-geometry (strict > 0.25), NO re-applied sigmoid (baked in),
    cxcywh->xyxy, un-letterbox, single-class GLOBAL NMS (IoU 0.45).
[x] Detection overlay (BoxFit.cover mapping, repaint-on-change) + HUD with live
    count, pipeline latency, and a buf/rot/img diagnostic line for device
    orientation confirmation.
[x] Tier A unit tests (9 named files, 10 cases) + A4 hot-path micro-benchmark.
[x] Tier A1 analyze (in worktree): `flutter analyze` -> No issues found (0/0).
[x] Tier A3 unit tests (in worktree): 10/10 pass.
[x] Tier A4 benchmark (in worktree): median 2.90ms, p90 3.00ms (JIT; post-proc
    budget only, excludes the NPU model.run).
[x] Tier B: applied + measured one structural optimization (inline-BGRA +
    format-hoist + reciprocal-multiply): 3.60ms -> 2.90ms median (~-19%, well
    over the 0.5% rule). Other levers applied by design (see Optimization log).
[x] Committed to branch app/vehicleplate (commit subject "VehiclePlateYOLO:
    implement on-device license-plate detector (GATE 3)").
[x] iOS deployment-target config: Podfile platform :ios, '16.6' + post_install
    IPHONEOS_DEPLOYMENT_TARGET 16.6, project.pbxproj 13.0 -> 16.6 (x3),
    NSCameraUsageDescription added to Info.plist.
[x] Android release config: minSdk 24, pinned AGP 8.9.1 / Kotlin 2.1.0 / Gradle
    8.11.1, android.suppressUnsupportedCompileSdk=36, isMinifyEnabled=false +
    isShrinkResources=false (R8 strips Melange JNI classes otherwise),
    useLegacyPackaging for the .so's.
[x] Tier A2 device-target release COMPILE (unsigned), both platforms GREEN:
    - iOS: `flutter build ios --release --no-codesign` -> pod install linked the
      vendored ZeticMLange.xcframework (device arm64), Xcode build done, built
      Runner.app (28.4MB).
    - Android: `flutter build apk --release` -> assembleRelease OK, built
      app-release.apk (196.1MB; large because minify is off, by design).
[ ] [BLOCKED - human, device-only] iOS signing identity (team WVJ22PPYBP) +
    physical-device RUN in RELEASE (debug hangs on recent iOS/Xcode). The
    unsigned compile passes; only codesigning + the hardware run remain.
[ ] Confirm served runtimeApType on the device console (expected NPU ~1.33ms;
    treat the console value as truth, not the dashboard number).

Deliverables
- Flutter source under apps/VehiclePlateYOLO/Flutter/ (lib: main, theme,
  screens/{loading,camera}, services/{melange_service [dedicated isolate],
  preprocessor, postprocessor, nms, letterbox, orientation, frame_data},
  models/detection, widgets/{detection_overlay, hud}).
- Tier A tests under Flutter/test/ (9 files, 10 cases) + test/benchmark/
  hot_path_benchmark.dart.
- Model assets (GATE 0): export.py, koushim-yolov8-license-plate.onnx,
  sample_input.npy, melange_upload.md, model_selection.md; registered Melange
  model ajayshah/VehiclePlateYOLO v1 (READY).

Build command (device)
  flutter run -d <UDID> --release --dart-define=ZETIC_KEY=<your_zetic_key>
(An empty ZETIC_KEY surfaces a clear on-screen error on the loading screen.)

Device console (watch for served backend + native crashes)
  xcrun devicectl device process launch --console --terminate-existing \
    --device <UDID> com.zeticai.vehicleplateyolo

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

Note on the mid-session macOS TCC revocation (now resolved)
  During the dark build, macOS revoked the terminal's Full Disk Access, blocking
  all read/write/git under ~/Desktop (CLAUDE.md section 5). Work continued
  against a validated scratchpad mirror; after access was restored the validated
  copy was reconciled into this worktree and the full Tier A battery was re-run
  here before commit. Nothing was lost. The iOS/Android signing config remains
  the only outstanding (human/device-only) work.
