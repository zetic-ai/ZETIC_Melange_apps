## Goal

A still-image, fully on-device dense retail-shelf SKU detector for Flutter (iOS +
Android), product name "ShelfSense", powered by a YOLO11s single-class detector
(trained on SKU-110K) through the ZETIC Melange SDK. The user picks a shelf photo
(file / gallery) or taps a bundled sample; the app letterboxes it to 640x640, runs
one on-device inference via Melange, decodes the channel-major [1,5,8400] output,
runs pure-Dart global NMS, and overlays a box on every product facing plus a
prominent "N products detected" count and an inference-latency readout. No camera,
no upload — one-shot image-in / boxes-out.

Status: READY FOR DEVICE (GATE 3). Tier A green, Tier B done, Tier C below.

## Todo List

- [x] Stage-0: select model (chistopat/sku110k-yolo11-object-detector, YOLO11s).
- [x] Stage-0: export YOLO11s -> shelfscan-yolo11s-sku110k.onnx + sample_input.npy.
- [x] Stage-0: validate pre/post pipeline over 36 SKU-110K shelves (median recall 0.903), select 3 demo frames (validate_demo.py, demo_images/).
- [x] Register model on Melange as ShelfScanYOLO v1 (GATE-0 CONFIRMED: exact name `ShelfScanYOLO`, version 1, RUN_AUTO; served images float32[1,3,640,640] -> output0 float32[1,5,8400]; model READY, full benchmark report present).
- [x] Scaffold Flutter project (zetic_mlange 1.8.1, image_picker, image, flutter_launcher_icons; iOS min 16.6, Android minSdk 24; 3 demo images bundled as one-tap samples).
- [x] Wire personalKey from GITIGNORED lib/config/secrets.dart (+ committed secrets.example.dart template). Confirmed git-ignored (git check-ignore) so the real key can never be committed; real key pasted into secrets.dart only.
- [x] Product display name "ShelfSense" (iOS CFBundleDisplayName, Android android:label, MaterialApp title, app-bar + loading text). Bundle id (com.zetic.shelfscanyolo), folder, and Melange model name unchanged.
- [x] models/detection.dart (BBox with clamped-area IoU + Detection).
- [x] services/melange_service.dart (create -> warm-up dummy inference -> run -> close; handle isolate-bound; RUN_AUTO; copies output out of native view).
- [x] services/preprocessor.dart (decode + EXIF bakeOrientation -> letterbox 640 gray-114 pad -> RGB -> /255 -> NCHW; records LetterboxTransform).
- [x] services/letterbox.dart (forward/inverse, integer floor-div pad — matches validate_demo.py exactly).
- [x] services/postprocessor.dart (channel-major [1,5,8400] decode; score AS-IS no re-sigmoid; strict >0.25 threshold-before-geometry; cxcywh->xyxy; inverse-letterbox to original px).
- [x] services/nms.dart (single-class GLOBAL NMS, IoU 0.45, suppress on strict iou>thr; flat-typed-array hot loop — Tier-B optimized).
- [x] services/display_fit.dart (original px -> BoxFit.contain displayed rect).
- [x] services/shelf_scanner.dart (pipeline compose + per-stage timings).
- [x] widgets/detection_overlay.dart (image + CustomPaint boxes, shared LayoutBuilder so display fit == draw fit).
- [x] widgets/hud.dart (headline count + latency; on-screen debug line — release builds swallow Dart print).
- [x] screens/loading_screen.dart (download progress + warm-up + error/retry).
- [x] screens/main_screen.dart (gallery pick + 3 sample chips, busy-guarded scan, overlay + HUD + debug toggle).
- [x] main.dart + theme.dart (ShelfSense dark "scan-green" identity).
- [x] assets/icon/app_icon.png (1024x1024 shelf + green detection boxes glyph); flutter_launcher_icons generated iOS AppIcon.appiconset + Android mipmaps.
- [x] Tier-A unit tests: channel-major decode, no-extra-sigmoid, coordinate space, threshold boundary, letterbox inverse round-trip (+ display fit), global NMS IoU 0.45 (incl. strict-boundary). 23 tests, all pass.
- [x] Integration harness on real demo images (preprocessing parity vs validate_demo.py letterbox) + synthetic dense decode+NMS pass.
- [x] test/benchmark/hot_path_benchmark.dart (mock-tensor A4 micro-benchmark).
- [x] flutter analyze clean (0 issues); iOS device release build compiles (no-codesign, Runner.app 32.6MB); Tier-B optimization pass with measured deltas (below).
- [ ] GATE 3 — human physical-device run (release, signed). Test device: UNKNOWN (to be assigned). This is the honest 70% an agent cannot verify; see Tier C.

## Deliverables

- Flutter source under apps/ShelfScanYOLO/Flutter/ (screens, MelangeService,
  preprocessor/letterbox, postprocessor, nms, display_fit, shelf_scanner,
  detection model, overlay + HUD, theme; 23 unit tests + A4 benchmark).
- Model assets (Stage 0): export.py, shelfscan-yolo11s-sku110k.onnx,
  sample_input.npy; registered Melange model ShelfScanYOLO v1 (READY).
- Bundled demo assets: 3 SKU-110K frames (research/non-commercial license flag —
  swap before any commercial context; see SPEC Licensing).
- Custom launcher icon (shelf + scan-green boxes) for iOS + Android; product
  display name "ShelfSense".
- Diagnostics: this HANDOFF.md, on-screen HUD/debug line, Tier-C checklist.

## Validation report (Tier A)

- A1 analyze: `flutter analyze` -> No issues found (0 errors, 0 warnings).
- A2 build: `flutter build ios --release --no-codesign` -> Built Runner.app
  (32.6MB); ZeticMLange pod linked. Custom icon + "ShelfSense" display name set.
  (Signed device build is the human's GATE-3 step.)
- A3 unit tests: 23/23 pass. Covers every THIS-model trap: channel-major
  [1,5,8400] decode (with a row-major distractor), score used as-is (asserted NOT
  the double-sigmoid value; float32 stores 0.30 as ~0.30000001, confirming
  as-is), pixel-space (~640) not normalized, strict >0.25 threshold boundary,
  letterbox inverse round-trip + BoxFit.contain display mapping, single-class
  GLOBAL NMS IoU 0.45 incl. strict > boundary, real-demo-image preprocessing
  parity vs validate_demo.py.
- A4 hot-path micro-benchmark (pure-Dart preprocess + decode + NMS; 1440x1080
  frame, ~2000 above-threshold overlapping anchors, n=40):
  median 48.7 ms (post-optimization). Stage split: preprocess ~39.7 ms
  (image copyResize-bound), decode ~0.03 ms (threshold-first), NMS ~9.4 ms.
  NOTE: this is the pure-Dart post-processing budget only, NOT end-to-end device
  latency (the NPU/CPU inference time is fixed by Melange and only visible on
  hardware).

## Tier B optimization log (0.5% rule; budget = A4 median)

- [APPLIED] NMS over flat Float32List x1/y1/x2/y2/area arrays instead of
  dereferencing Detection.box.* objects in the O(n^2) loop; pre-sort once,
  pre-compute areas, early-continue on no-overlap. Before 51.4 ms -> after
  48.7 ms total (~2.7 ms, ~5.3% of budget; NMS stage ~12.1 -> ~9.4 ms). Kept.
- [APPLIED, already in design] Threshold-before-geometry in decode: rejected
  anchors cost ~nothing -> decode ~0.03 ms on 8400 anchors. Kept.
- [APPLIED, already in design] Single fused pass for /255 + NCHW over the RGB
  byte buffer (no intermediate buffers, typed-data views). Kept.
- [REJECTED] Byte->float /255 lookup table: measured ~0.1 ms delta (< the 0.26 ms
  = 0.5% bar; normalization is not the bottleneck, copyResize is). Removed per
  the 0.5% rule to avoid complexity for no gain.
- [N/A — still-image, one-shot] Per-frame isolate reuse, _busy frame-drop
  throttle, cheapest-camera-pixel-format, repaint-only-on-change per frame:
  these are live-camera levers; ShelfSense runs one inference per user tap, not a
  video loop. A _busy guard still prevents re-entrant scans. Inference runs
  synchronously on the handle-owning (main) isolate because the Melange handle is
  isolate-bound; if the preprocess block causes visible jank on a big photo, the
  future lever is to move decode+letterbox into a compute() isolate (copies
  ~1 MB in / ~4.9 MB out) — deferred until a device shows it is needed.
- [APPLIED] Model lifecycle: warm-up dummy inference right after create() so the
  first real scan is not the cold one; SDK caches the model (not re-downloaded).

## Tier C — human-handoff runtime-risk checklist (surfaced, not testable here)

- Served artifact: client cannot force backend/precision. Expect a realistic
  non-crashing fallback of TFLITE_FP16 / CPU (hundreds of ms), NOT necessarily
  the dashboard's NPU 5.43 ms row. Read the ACTUAL served target + apType from
  the native console (`runtimeApType=...`) — that is truth. Same YOLO11s family
  as PyroGuard, so the FP32-GPU MPSGraph crash path (iOS/macOS 26.3+) is possible
  on a new OS; if it SIGABRTs at first inference, escalate to ZETIC to filter GPU
  for that OS (not client-fixable; no modelMode avoids it).
- modelMode: RUN_AUTO. Do not treat it as a crash workaround.
- Native observability: watch
  `xcrun devicectl device process launch --console --terminate-existing --device <UDID> com.zetic.shelfscanyolo`
  Dart print/debugPrint does NOT surface in a release console — on-device
  diagnostics are on the app's DEBUG HUD line (scale/pad, in/out shapes,
  per-stage ms, raw first box, pre/post-NMS counts). Toggle via the app-bar bug
  icon.
- Signing / OS gates (manual): signing identity/team, Developer Mode, "Always
  Allow", iOS >= 16.6. iOS SIMULATOR IS A DEAD END (device-only ios-arm64 slice);
  every iteration is a signed RELEASE device build (debug hangs on launch on
  recent iOS/Xcode).
- Build config: release on device (Runner.app already builds no-codesign; add
  signing team to deploy). Android minSdk 24.
- Network / cold start: model downloads on first launch; on poor Wi-Fi that is a
  spinner (progress shown). Rehearse a fresh install; consider pre-warm.
- Non-determinism: server-side selection can change artifact run-to-run. "It ran
  once" is not evidence — require multiple clean cold starts + one fresh install,
  and re-verify after any backend/model re-target.
- End-to-end accuracy check (the real integration test): on device, load each
  bundled sample and confirm detected counts track the measured demo numbers
  (491 / 221 / 159) — this cannot be reproduced in a pure-Dart unit test because
  the ONNX model does not run there; the Dart pre/post is proven parity-correct
  vs validate_demo.py, so a count mismatch on device points at the served
  artifact, not the pipeline.
- Secrets: the personal key is embedded in the client (gitignored secrets.dart).
- If too slow on the target device: drop-in lighter same-repo YOLO11n
  (weights/sku110k-yolo11-n640.pt) at identical 640 / 1-class / [1,5,8400] — no
  pipeline changes.

## References

- App directory: apps/ShelfScanYOLO (Flutter under apps/ShelfScanYOLO/Flutter)
- Spec: apps/ShelfScanYOLO/SPEC.md (+ model_selection.md, demo_images/DEMO_IMAGES.md)
- Core SDK: ZETIC Melange (zetic_mlange 1.8.1, Flutter FFI)
- Model: YOLO11s SKU-110K single-class — chistopat/sku110k-yolo11-object-detector
  (input float32[1,3,640,640] NCHW RGB /255 letterboxed; output float32[1,5,8400]
  channel-major [cx,cy,w,h,object_score], score pre-sigmoid'd, box in 640 px
  space, NMS in Dart; class: object). Registered Melange name ShelfScanYOLO v1.
- Frameworks: Flutter 3.44 / Dart 3.12, image_picker, image (decode/EXIF/resize),
  CoreML / Apple Neural Engine or CPU fallback (via Melange), Ultralytics (export)
- Reference implementation: apps/FireDetectionYOLO (PyroGuard) — structure,
  Melange lifecycle, pre/post/NMS split, iOS config. Key difference: ShelfSense
  is still-image upload (no camera, no camera-orientation trap) — only EXIF +
  letterbox->displayed rect geometry.
- Bundle id: com.zetic.shelfscanyolo · iOS min 16.6 · Android minSdk 24
- Test device: UNKNOWN (to be assigned by human for the GATE-3 device run).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
