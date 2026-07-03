## Goal

A fully on-device dental-radiograph pathology analyzer for Flutter (iOS primary,
Android secondary), product name "OraLens", powered by a YOLO11n detector
(liodon-ai/dental-panoramic-detector, CC-BY-NC-4.0) through the ZETIC Melange SDK.
This is a STILL-IMAGE UPLOAD app (no live camera): the user picks an X-ray from the
library or taps one of three bundled sample radiographs, the app runs single-shot
on-device inference at 640x640, and overlays labeled caries / periapical_lesion /
impacted_tooth boxes with a per-class count and an inference-latency readout. A
required, always-visible non-diagnostic disclaimer states this is a research
capability proof, not a diagnostic device, and that on-device deployment changes
data-residency only and does NOT imply or confer FDA clearance.

Registered Melange model name is `DentalXRayDetect` (capital "R"), under the ZETIC
org — this DIFFERS from the app folder name `DentalXrayDetect` (lowercase "r"). The
SDK `ZeticMLangeModel.create(name: ...)` passes `DentalXRayDetect` exactly.

Status: READY FOR DEVICE (Tier A green, Tier B satisfied). The worker cannot run a
physical device; live inference is the human's GATE-3 step.

## Todo List

- [x] Stage 0: export YOLO11n (liodon-ai/dental-panoramic-detector, best.pt) to ONNX (opset 12, static axes, imgsz 640) as dentalxray-yolo11n.onnx + sample_input.npy.
- [x] Stage 0: register on Melange as ZETIC | DentalXRayDetect, version 1, RUN_AUTO, READY (full benchmark report present). Served I/O: input images float32[1,3,640,640] NCHW RGB 0..1; output output0 float32[1,7,8400] channel-major, sigmoid baked, NMS not baked.
- [x] Stage 0: validate 3 demo radiographs (val_28 / val_38 / val_32) against DENTEX ground truth at conf 0.45, per-class NMS IoU 0.45 (see demo_images/DEMO_IMAGES.md).
- [x] Create core Flutter structure (loading_screen, main_screen, theme, HUD, always-on disclaimer banner).
- [x] Model constants: kInputSize = 640 (guarded by a test), kNumAnchors = 8400, kNumChannels = 7, kNumClassChannels = 3, conf 0.45, per-class NMS IoU 0.45; 3 labels {0 caries, 1 periapical_lesion, 2 impacted_tooth} + per-class colors.
- [x] MelangeService + long-lived inference isolate (create -> warm-up -> run -> close), model created and run INSIDE the isolate (SDK binds the native handle to the creating isolate). name "DentalXRayDetect" EXACTLY, modelMode RUN_AUTO, personalKey from gitignored lib/config/secrets.dart.
- [x] Preprocessing: decode picked/sample image off UI isolate (image pkg) + bake EXIF orientation; decode straight to packed RGB (grayscale luma auto-replicates to 3 channels); BILINEAR letterbox 640 (img.copyResize Interpolation.linear, matching the validate_demo.py / Ultralytics cv2.INTER_LINEAR harness) + gray pad (114/255 = 0.447, integer floor((640-nw)/2) centering) + normalize/255 + NCHW into a pre-allocated buffer; record r, padX, padY.
- [x] Post-processing: decode [1,7,8400] CHANNEL-MAJOR (stride across 8400, not the 7), max-of-3-class + argmax label, NO re-sigmoid, threshold BEFORE box geometry (strict `>`, pinned to the harness), cxcywh -> xyxy in 640 px, inverse letterbox to original-image pixels + clamp, per-class NMS IoU 0.45.
- [x] Still-image input flow: image_picker gallery pick + 3 bundled sample radiographs loaded from assets (no photo-library add prompt) for a one-tap demo.
- [x] Detection overlay: single BoxFit.contain transform (letterbox inverse -> original-image space -> contain-fit display), no rotation; class color + label + confidence; InteractiveViewer pinch-zoom with constant on-screen stroke width.
- [x] HUD: per-class counts (all 3 classes), inference latency (total + pre/run/post), output tensor shape + raw first box — surfaced on the UI (release-build Dart print does NOT reach the native console).
- [x] Always-visible non-diagnostic disclaimer banner, pinned to the bottom of both the loading and analyzer screens (every state).
- [x] Tier A unit tests: channel-major decode (+ transposed-read guard), max-of-3-class, no-double-sigmoid, threshold boundary (strict > @ 0.45), coordinate-space, letterbox inverse round-trip, per-class NMS both-survive + same-class collapse, still-image contain-fit round-trip, + a demo-image integration test that runs a real bundled radiograph (val_28) through the bilinear preprocess. 21 tests, all green.
- [x] Tier A: flutter analyze (0 issues); iOS release build compiles + links the vendored ZeticMLange.xcframework (Runner.app 42.5MB, --no-codesign).
- [x] A4 hot-path micro-benchmark (640 preprocess + 8400 decode + per-class NMS on mock tensors of the real shape) + Tier B optimization log (0.5% rule, see below).
- [x] Custom domain-identifying launcher icon (molar in a teal detection box, 1024x1024 -> flutter_launcher_icons, iOS + Android) + product name "OraLens" as user-facing display name (iOS CFBundleDisplayName, Android android:label, MaterialApp title).
- [x] iOS config: iOS 16.6 min (Podfile + pbxproj IPHONEOS_DEPLOYMENT_TARGET), NSPhotoLibraryUsageDescription for the picker; Android minSdk >= 24.
- [x] Personal key wired via gitignored lib/config/secrets.dart (real key in place, verified untracked) + committed secrets.example.dart template with a placeholder.
- [ ] **[BLOCKED – human/device]** Live on-device inference (model.run) + served-artifact backend (target + apType from native console) + end-to-end latency. Owner: human. The create/warm-up/run path is written, compiles, and links, but model.run only executes on a physical device (iOS simulator is a dead end: device-only xcframework slice). Confirm the served artifact is NOT FP32-GPU CoreML on iOS/macOS 26.3+ (MPSGraph SIGABRT trap); budget CPU-speed fallback until runtimeApType=NPU is confirmed on hardware.
- [ ] iOS signed device run (signing identity is a human gate; see Tier C).
- [ ] Android run verification once iOS is stable (pipeline is platform-agnostic; the still path has no camera/YUV code to exercise).
- [ ] On-device demo-reproduction check: confirm val_28/38/32 overlays match the measured DENTEX detections. Preprocess now resamples BILINEAR to match the harness/Ultralytics, so confidences should track the validated numbers closely (any residual gap is the served backend precision, not resampling).

## Deliverables

- Flutter source under apps/DentalXrayDetect/Flutter/lib: main, screens/
  (loading_screen, main_screen), services/ (melange_service, inference_isolate,
  preprocessor, postprocessor, nms, image_decoder, samples), models/ (detection,
  label), widgets/ (detection_overlay, coordinate_mapping, disclaimer_banner),
  config/ (secrets.example.dart committed; secrets.dart gitignored), theme.dart.
- Tests under Flutter/test: decode_test, threshold_test, letterbox_test, nms_test,
  still_mapping_test, benchmark/hot_path_benchmark. 20 unit tests green.
- Model assets (Stage 0): export.py, dentalxray-yolo11n.onnx, sample_input.npy,
  model_selection.md, melange_upload.md, validate_demo.py; registered Melange model
  ZETIC | DentalXRayDetect v1 (READY).
- Bundled demo radiographs (val_28 / val_38 / val_32) in Flutter/assets/samples.
- Custom launcher icon (Flutter/assets/icon/app_icon.png + generator tool/gen_icon.dart)
  and "OraLens" product display name.
- iOS config (Info.plist photo-library usage + OraLens display name, Podfile iOS 16.6,
  pbxproj target 16.6) + Android (android:label OraLens, minSdk 24).

## Tier A results

- flutter analyze: No issues found (0 errors, 0 warnings).
- Unit tests: 21/21 pass (decode x5, threshold x3, letterbox x4, nms x4, still-mapping x4,
  demo-image integration x1).
- Build: iOS release --no-codesign compiles and links zetic_mlange xcframework;
  Runner.app 42.5MB. (Signed device build is the human signing gate.)
- A4 hot-path micro-benchmark (pure-Dart: RGB 640 BILINEAR preprocess + 8400 decode +
  per-class NMS; mock tensors of the real shape; host Dart VM, 120 iters): median
  6.93 ms, p90 8.04 ms, min 6.70 ms, max 8.35 ms. The jump from the earlier 0.82 ms
  nearest-neighbour baseline is the bilinear resize, adopted deliberately for harness
  fidelity (see Tier B). This runs ONCE per uploaded image (still app), not per frame,
  and still excludes the device-only model.run() — it is the post-processing budget, not
  end-to-end latency.

## Tier B optimization log (0.5% rule; budget ~6.9 ms)

- Threshold-before-geometry (MEASURED): decode median 0.472 ms (threshold-after) ->
  0.037 ms (threshold-before) = 0.435 ms saved. Far exceeds 0.5%.
- FIDELITY OVER SPEED (deliberate): the resize step uses BILINEAR (img.copyResize
  Interpolation.linear) instead of a fused nearest-neighbour single pass, to reproduce
  the validated harness (cv2.INTER_LINEAR) so borderline detections at the 0.45 gate
  match the measured recall. Cost: ~+6 ms preprocess per upload. Justified because this
  is a still app (one inference per user tap, NOT a per-frame live loop), so accuracy
  parity with the validated numbers outweighs the one-shot preprocess cost.
- Pre-allocated NCHW input Float32List reused across inferences (applied): avoids a
  ~4.9 MB (3*640*640*4) allocation per image.
- Typed-data views throughout (Uint8List/Float32List), no boxed Lists (applied).
- Decode allocates no Detection until the confidence gate passes (applied).
- NMS: bucket per class, sort once per bucket, box areas via getter, threshold before
  the O(n^2) step so n is tiny (applied).
- _busy single-in-flight guard (applied; still app = guards double-tap).
- Warm-up dummy inference right after create; SDK caches the model (applied).
- Overlay repaints only when image/detections/zoom change (shouldRepaint identity)
  (applied).
- Dedicated long-lived isolate; image bytes in via TransferableTypedData (ownership
  transfer, no copy), detections out (applied).

## Tier C runtime-risk checklist (human, device-only — surfaced, not tested)

- Served artifact: expect FP32; RUN_AUTO picks a Neural-Engine artifact if/when ZETIC
  serves one, else CPU (TFLITE_FP16, ~100+ ms) as the realistic fallback. The client
  cannot force it. Read the ACTUAL served target+apType from the native console
  (e.g. runtimeApType=CPU) — that is the truth, not the benchmark row.
- Known crash path: FP32-GPU CoreML can SIGABRT in Apple MPSGraph on iOS/macOS 26.3+ on
  the FIRST inference (uncatchable in Dart). No client modelMode avoids it; the durable
  fix is ZETIC filtering GPU server-side for the affected OS. Confirm the served apType
  is NOT GPU on affected OS versions; escalate to ZETIC if it is.
- modelMode: RUN_AUTO. Do NOT treat any mode as a crash workaround.
- Native observability: launch with
  `xcrun devicectl device process launch --console --terminate-existing --device <UDID> ai.zetic.dentalxraydetect`
  Dart print/debugPrint does NOT surface in this console on a release build — on-device
  diagnostics (latency, tensor shape, first box) are shown on the app HUD instead.
- Signing / OS gates (manual): signing identity/team, Developer Mode, "Trust", iOS 16.6
  min. iOS simulator is a dead end (device-only xcframework slice); use a physical device
  in RELEASE (debug hangs on launch on recent iOS/Xcode).
- Network / cold start: the model downloads on first launch; on poor conference Wi-Fi
  that is a spinner. Pre-download / pre-warm and rehearse a fresh install.
- Non-determinism: server-side selection can return a different artifact run to run.
  Acceptance = runs cleanly across multiple cold starts + at least one fresh install
  before it counts as demo-ready; re-verify after any backend/model re-target.
- Secrets: the ZETIC personal key is embedded in the client (lib/config/secrets.dart,
  gitignored).

## References

- App directory: apps/DentalXrayDetect
- Core SDK: ZETIC Melange (zetic_mlange 1.8.1, Flutter FFI) — API used:
  ZeticMLangeModel.create(personalKey, name, version, modelMode, onProgress) /
  model.run(List<Tensor>) / Tensor.float32List(data, shape:) /
  outputs.first.asFloat32List() / model.close().
- Model: YOLO11n dental pathology — liodon-ai/dental-panoramic-detector
  (input float32[1,3,640,640], output float32[1,7,8400] = 4 box + 3 class, sigmoid
  BAKED, NMS not baked; classes: caries, periapical_lesion, impacted_tooth; conf 0.45,
  per-class NMS IoU 0.45). License CC-BY-NC-4.0 (non-commercial; demo only). MIT drop-in
  fallback: Sentoz/dental-opg-cavity-detection-model.
- Reference implementations: apps/FireDetectionYOLO (PyroGuard), sibling YOLO worktrees
  AerialDetectYOLO (still+camera) and VehiclePlateYOLO (still-image upload UI).
- Frameworks: Flutter 3.44.3 / Dart 3.12.2, image_picker, image (EXIF decode),
  CoreML / Apple Neural Engine via Melange, Ultralytics (export).
- Test device: TBD by human (sibling apps ran iPhone 15 / A16, iOS 26.5).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
