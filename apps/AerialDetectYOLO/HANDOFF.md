## Goal

A real-time, fully on-device aerial / drone object-detection demo for Flutter (iOS primary, Android secondary), powered by a VisDrone-trained YOLOv8s detector (ENOT-AutoDL/yolov8s_visdrone, Apache-2.0) through the ZETIC Melange SDK. Streams the live rear-camera feed, runs detection each frame on a dedicated inference isolate at 928x928, and overlays labeled boxes (10 VisDrone classes) with a live per-class count + inference-latency HUD.

## Todo List

- [x] Create core Flutter structure (loading screen, camera screen, theme, HUD).
- [x] Adopt model: ENOT-AutoDL/yolov8s_visdrone (baseline_enot, imgsz 928) exported to ONNX (opset 12, static shapes), registered on Melange (ajayshah/AerialDetectYOLO, v1).
- [x] RESOLVE the sigmoid discrepancy (GATE-2 #1 risk): inspected the exported ONNX graph — output0 = Concat(axis=1)[box(Mul_2), /model.22/Sigmoid(class head)] — AND checked a real onnxruntime output range (class ch 4..13 in [0.0, 2.6e-4], box ch 0..3 in [2.8, 925]). Sigmoid IS BAKED IN. Decode does NOT re-apply it; class channels are treated as final probabilities. Guarded by test_no_double_sigmoid.
- [x] Melange lifecycle wrapper (create -> warm-up -> Tensor.float32List -> run -> close), created and run INSIDE a long-lived dedicated inference isolate so the 928 preprocess + ~247K-float decode never block the UI isolate.
- [x] Preprocessing: letterbox 928x928 (pad 114/255), fused single-pass resize+normalize+NCHW; iOS BGRA8888 swizzle inline, Android YUV420 path.
- [x] Post-processing: decode [1,14,17661] channel-major (stride across anchors), max over class channels 4..13, threshold > 0.25 BEFORE box geometry, cxcywh -> xyxy, inverse letterbox into source pixels, per-class NMS (IoU 0.45).
- [x] Detection overlay (BoxFit.cover, no rotation per the PyroGuard lesson) + HUD (latency, per-class counts, on-screen buffer WxH + raw box debug line).
- [x] personalKey via String.fromEnvironment('ZETIC_KEY') (--dart-define), never committed; clear error if empty.
- [x] iOS config: NSCameraUsageDescription, iOS 16.6 min (Podfile + pbxproj); Android: CAMERA/INTERNET perms, minSdk 24.
- [x] Tier A green (pure-Dart, no device): flutter analyze 0 errors/0 warnings; 15 unit tests pass; iOS release build (--no-codesign) compiles & links the zetic_mlange xcframework (Runner.app 28.7MB).
- [x] A4 hot-path micro-benchmark + Tier B optimization log (see Deliverables).
- [ ] **[BLOCKED – Melange model OPTIMIZING/awaiting READY]** Live on-device inference (model.run). The model was still OPTIMIZING at GATE-0 paste-back; the human build-unblocked the pure-Dart battery only. The create/warm-up/run code path is written and compiles but has NOT been exercised against a live READY model. PARKED at model.run() until the dashboard shows READY. Do not treat live inference, served-artifact backend, or end-to-end latency as verified.
- [ ] iOS signed device run (signing identity is a human gate; see Tier C).
- [ ] Android run verification once iOS is stable (YUV420 path is implemented but unexercised on hardware).
- [ ] On-device orientation confirmation: HUD prints buffer WxH; confirm no rotation is needed (mirrors PyroGuard) or adjust the overlay transform.

## Deliverables

- Flutter source under apps/AerialDetectYOLO/Flutter/lib (main, screens/, services/ [melange_service, inference_isolate, preprocessor, postprocessor, nms], models/ [detection, label], widgets/ [detection_overlay, hud, coordinate_mapping]).
- Tests under Flutter/test: decode_test, letterbox_test, nms_test, orientation_test, benchmark/hot_path_benchmark.
- Model assets: export.py, aerialdetect-yolov8s-visdrone.onnx, sample_input.npy, model_selection.md, melange_upload.md; registered Melange model (ajayshah/AerialDetectYOLO v1).
- iOS config: Info.plist camera usage, Podfile (iOS 16.6), pbxproj IPHONEOS_DEPLOYMENT_TARGET 16.6. Android: manifest perms, minSdk 24.

## Tier A results

- flutter analyze: No issues found (0 errors, 0 warnings).
- Unit tests: 15/15 pass (decode x5, letterbox x3, nms x4, orientation x3).
- Build: iOS release --no-codesign compiles and links zetic_mlange; Runner.app 28.7MB. (Signed device build is the human signing gate.)
- A4 hot-path micro-benchmark (pure-Dart: BGRA 928 preprocess + 17661 decode + per-class NMS; mock tensors of the real shape; host Dart VM, 120 iters): median 2.48 ms, p90 2.56 ms. NOTE: this excludes the device-only model.run(); it is the post-processing budget, not end-to-end latency.

## Tier B optimization log (0.5% rule; budget ~2.6 ms)

- Threshold-before-geometry (MEASURED): decode median 1.578 ms (threshold-after) -> 0.570 ms (threshold-before) = ~1.0 ms / ~64% decode saved, ~40% of the hot path. Far exceeds 0.5%.
- Pre-allocated input Float32List reused across frames (applied): avoids a ~10 MB (3*928*928*4) allocation every frame.
- Fused resize+normalize+NCHW single pass, BGRA->RGB swizzle inline (applied): no intermediate RGB buffer on the iOS hot path.
- Typed-data views throughout, no boxed Lists (applied).
- Cheapest camera format requested (BGRA iOS / YUV420 Android) (applied).
- Decode allocates no Detection until the threshold passes (applied).
- NMS: bucket per class, sort once per bucket, areas via getter, threshold before the O(n^2) step (applied).
- _busy frame-guard drops frames rather than queueing (applied).
- Warm-up dummy inference right after create; SDK caches the model (applied).
- Overlay repaints only when detections change (shouldRepaint identity) (applied).
- Dedicated long-lived isolate; frame bytes in via TransferableTypedData (ownership transfer), detections out (applied).

## References

- App directory: apps/AerialDetectYOLO
- Core SDK: ZETIC Melange (zetic_mlange 1.8.1, Flutter FFI) — verified API surface: ZeticMLangeModel.create(personalKey,name,version,modelMode) / model.run(List<Tensor>) / Tensor.float32List / outputs.first.asFloat32List() / close().
- Model: YOLOv8s VisDrone — ENOT-AutoDL/yolov8s_visdrone baseline_enot (input float32[1,3,928,928], output float32[1,14,17661] = 4 box + 10 class, sigmoid BAKED, no NMS; classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor).
- Frameworks: Flutter 3.44.3 / Dart 3.12.2, camera plugin, CoreML/ANE via Melange, Ultralytics (export).
- Test device: TBD by human (reference sibling app ran iPhone 15 / A16, iOS 26.5).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
