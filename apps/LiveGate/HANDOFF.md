# HANDOFF — LiveGate (TrueFace)

> Status: **GATE 3 — ready for device.** Tier A all green (analyze clean, both release builds pass, 19 unit tests + golden). Tier B logged. Tier C filled below. Both Melange models are still name/version placeholders pending the GATE-0 paste-back — injection is a one-file, one-commit change; the device run requires it. The worker claims "ready for device," not "done."

## Goal

A fully on-device KYC live-selfie gate for Flutter (iOS + Android), powered by two ZETIC Melange models: a MiniFASNet-V2 anti-spoof classifier (PAD) and an SFace ArcFace face-embedding model. Point the front camera at a face; the app detects the face app-side (ML Kit bbox + 5 landmarks), decides **LIVE vs SPOOF** (softmax class 1 ≥ 0.45), and for a real face returns a **1:1 cosine match** against a locally-enrolled reference vector (≥ 0.363). A SPOOF fails the gate regardless of match and its score is withheld; only LIVE + MATCH passes. No face bytes ever leave the device, and enrollment stores only the 128-d vector.

## Todo List

- [x] Stage 0: export + validate both ONNX (pad_minifasnet_v2.onnx [1,3,80,80]→[1,3]; face_sface.onnx `data`[1,3,112,112]→`fc1`[1,128]); demo goldens reproduced (PAD live=1.000, ÷255=0.006).
- [x] Create Flutter app skeleton under apps/LiveGate/Flutter/ per CLAUDE.md §4 anatomy (from FireDetectionYOLO reference).
- [x] `lib/services/model_registry.dart`: the ONLY file referencing Melange names/versions — both models as clearly-marked placeholders (name/version ONLY; no key).
- [x] `lib/config/secrets.example.dart` (tracked, `<ZETIC_PERSONAL_KEY>` template) + gitignored `lib/config/secrets.dart` for the real key — RetinaDRScreen convention.
- [x] `lib/services/melange_service.dart`: create/warm/run/close BOTH models (PAD + FACE) on the main isolate; dummy warm-up inference each.
- [x] Face-detect stage: integrate google_mlkit_face_detection (on-device, bundled, offline) → bbox + 5 landmarks. Repo MediaPipe folders (`apps/MediaPipe-Face-Detection`, `apps/MediaPipe-Face-Landmarker`) inspected and REJECTED as native-only demos (Kotlin/Swift, no Flutter/Dart to reuse; Landmarker is a 468-pt mesh) — reusing them would mean 4 Melange loads + porting SSD/mesh decode to Dart.
- [x] `lib/services/preprocessor.dart`: BGR conversion; PAD 2.7× clip-shift crop → 80×80 BGR [0,255] NCHW; runs in compute() isolate.
- [x] `lib/services/face_align.dart`: 5-point similarity/affine warp to ArcFace 112×112 template → BGR [0,255] NCHW.
- [x] `lib/services/gate.dart`: PAD softmax + class-1 liveness threshold; SFace L2-norm + cosine; gate composition (SPOOF → reject, hide score).
- [x] `lib/services/enrollment.dart`: enroll → store L2-normalized 128-d vector in flutter_secure_storage (Keychain/Keystore); no image persisted.
- [x] UI: front-camera preview + face box; big LIVE/SPOOF verdict; match score + PASS/FAIL badge; enroll action; "on-device · no cloud" badge; per-stage latency + buffer WxH HUD (no Dart print on release).
- [x] Tier A tests (19 tests incl. a Python-generated golden preprocessing fixture) — all green.
- [x] `test/benchmark/hot_path_benchmark.dart`: mock-tensor micro-benchmark of the pure-Dart hot path; median baseline recorded.
- [x] Tier B optimization pass with measured deltas (≥0.5% rule).
- [x] Custom launcher icon (shielded face + verified glyph) via flutter_launcher_icons (iOS + Android, remove_alpha_ios).
- [x] Product name **TrueFace** as display name (iOS CFBundleDisplayName, Android android:label, in-app title); folder/bundle-id/Melange names unchanged.
- [x] Android release gauntlet per LESSONS.md (AGP 8.9.1 / Kotlin 2.1.0 / Gradle 8.11.1, minify off, INTERNET + ACCESS_NETWORK_STATE, proguard keeps, legacy jniLibs packaging).
- [x] iOS release build config (iOS 16.6, NSCameraUsageDescription, `flutter build ios --release --no-codesign` passes).
- [x] `flutter analyze` clean; Tier A green; Tier C filled.
- [ ] **[BLOCKED – human/ZETIC dashboard]** GATE-0 paste-back: registered names/versions + served shapes for `ajayshah/LiveGatePAD` and `ajayshah/LiveGateFace`. Until then model_registry.dart holds placeholders. Root cause: model registration is the one manual dashboard step; injection is a one-file, one-commit change.
- [ ] **[BLOCKED – human device]** Physical-device run (requires the paste-back injection + a real personal key in secrets.dart): confirm served `runtimeApType`, buffer WxH/orientation, ML Kit rotation reconciliation, and live spoof separation.

## Deliverables

- Flutter source under apps/LiveGate/Flutter/ (two-model MelangeService, ML Kit face detect, frame→upright BGR, PAD preprocessor, ArcFace face_align, gate, enrollment, screens, widgets/HUD).
- Model assets (Stage 0, committed): export.py, pad_minifasnet_v2.onnx, face_sface.onnx, sample_input_pad.npy, sample_input_face.npy.
- lib/services/model_registry.dart — single late-binding file for both model name/version placeholders (no key); lib/config/secrets.example.dart template + gitignored secrets.dart for the key.
- Tier A test suite (19 tests) + Python-generated golden fixture (test/fixtures/pad_golden.json); hot-path micro-benchmark.
- Reproducible Python tools: tools/gen_golden.py (golden), tools/e2e_pipeline_check.py (real-ONNX PAD fidelity + aligned FACE separation).
- Custom launcher icon (assets/icon/app_icon.png) + TrueFace product name.

## Build Plan

- **Pipeline (per frame, `_busy`-guarded):** copy camera planes → FrameData; run ML Kit face detection (native, async) on the frame → best bbox + 5 landmarks; if no face, HUD "no face" and release. In a compute() isolate: frame→upright BGR once, build the PAD tensor (2.7× clip-shift crop → 80×80 BGR [0,255] NCHW) and the FACE tensor (5-pt warp → 112×112 BGR [0,255] NCHW). On the main isolate run PAD → softmax → liveScore. **If LIVE and enrolled**, run FACE → L2-norm → cosine vs enrolled (FACE `run` is skipped on SPOOF/not-enrolled — saves the 38 MB inference and matches the gate). Compose verdict; setState HUD.
- **Threading:** heavy pure-Dart (BGR convert, crop, warp) in compute() isolates; both `model.run` calls on the main isolate (handle bound to it); ML Kit runs on its own native threads.
- **Late-binding:** model_registry.dart exposes `pad` and `face` as `ModelRef(name, version)` placeholders ONLY (no key). The personal key follows the repo secrets convention (tracked `lib/config/secrets.example.dart` + gitignored `lib/config/secrets.dart`). melange_service.dart is the sole consumer. Thresholds (0.45, 0.363) live in gate.dart.

## Validation report — Tier A results

- **A1 analyze:** `flutter analyze` → No issues found (0 errors, 0 warnings).
- **A2 build:** iOS `flutter build ios --release --no-codesign` → Built Runner.app (68.4 MB). Android `flutter build apk --release` → app-release.apk (213 MB). Both pass; only the expected AGP/Kotlin deprecation warnings (pins are the LESSONS.md-proven baseline). Custom launcher icon generated (iOS AppIcon.appiconset + Android mipmaps). Display name TrueFace on both.
- **A3 unit tests:** 19 tests, all green — T1 ÷255-saturation guard (BGR order + range >1.0), T2 live-class index (softmax[1]), T3 2.7× crop clip-SHIFT (centered / edge-shift / scale-cap), T4 FACE BGR+[0,255] no-norm, T5 L2-norm semantics (unit·unit=1, cosine∈[-1,1], raw dot≠cosine), T6 alignment (template→template identity + recovers a known similarity), T7 both thresholds (0.45, 0.363 boundaries), T8 gate composition (SPOOF never passes, score withheld), T9 orientation (0/90/180 transforms), T10 golden fidelity (Dart == Python NCHW).
- **A4 micro-benchmark:** pure-Dart hot path median (720×1280 mock frame, `flutter test` JIT): baseline 4246 µs, optimized 4188 µs. AOT on device is faster; the compute() isolate hop is excluded (this is the post-processing budget, not end-to-end).
- **E2E fidelity (Python, real committed ONNX — `tools/e2e_pipeline_check.py`):** PAD live_crop softmax[1]=**1.000**, spoof_crop=**0.000**, ÷255 path=**0.006** (saturated — proves feed [0,255]). FACE, YuNet 5-pt detect + the same similarity warp as face_align.dart: same-person (Bush/Bush)=**0.685**, different-person (Bush/Powell)=**0.075** → the 0.363 threshold **separates cleanly**. Naive resize WITHOUT alignment gives different-person=**0.441** → a FALSE MATCH; this is the concrete reason 5-pt alignment is mandatory, not optional.

## Tier B optimization log

- **Pre-allocated input buffers** (reuse `out:` Float32List across frames instead of allocating per frame): median 4246→4188 µs = **+1.4%** (> 0.5% → KEPT).
- **Single-pass fusion** (resize + BGR sample + NCHW layout done in one pass over the output grid in buildPadInput/buildFaceInput; upright-BGR built once and shared by both branches): structural, no intermediate buffers — no separate before/after number.
- **Typed-data views throughout** (Float32List / Uint8List, no boxed `List<double>` in the hot path): structural.
- **Skip the FACE `model.run` on SPOOF / not-enrolled:** avoids the 38 MB SFace inference on every non-live frame — the single biggest device-time saver, but it lives in the NPU-bound path so it cannot be shown on the CPU micro-benchmark.
- **Warm-up dummy inference per model after load:** the first live frame is not the cold one.
- **`_busy` frame-drop guard:** one frame in flight at a time; frames are dropped, not queued.
- **Justified skips:** no per-frame isolate spawn (a single compute() hop builds both tensors); overlay repaint gated by `shouldRepaint`.

## Runtime Risk (Tier C) — surfaced for the human device run

- **Served artifact.** Requested RUN_AUTO for both models; the client cannot force the backend. Expect a CPU fallback (TFLITE_FP16, ~hundreds of ms) until ZETIC serves a Neural-Engine artifact. Read the actual served `target`+`apType` (`runtimeApType=…`) from the native console — that, not the dashboard, is ground truth. The iOS/macOS 26.3+ CoreML-GPU MPSGraph crash is handled server-side by ZETIC filtering GPU; if a new OS crashes in MPSGraph, escalate to ZETIC.
- **modelMode.** RUN_AUTO on both. Do NOT expect a modelMode to steer off a crashing artifact.
- **Cold start / network.** TWO Melange models download on first launch (~1.8 MB PAD + ~38.5 MB FACE). ML Kit face detection is bundled/offline, but Melange needs network on first launch — so the "on-device · no cloud" badge is honest at inference time, not during the first-launch download. Pre-warm / rehearse a fresh install on the venue Wi-Fi.
- **Native observability.** Watch `xcrun devicectl device process launch --console --terminate-existing --device <UDID> com.zeticai.livegate`. Dart print/debugPrint does NOT reach the console in release — all diagnostics (per-stage latency, buffer WxH) are on the HUD.
- **Orientation / ML Kit reconciliation (worker-owned risk).** The buffer rotation handed to ML Kit MUST match `frameToUprightBgr` so the detector's box/landmarks and the model crops share one upright space; landmarks are ordered by image-x (mirror-robust). Chosen rotation: iOS 0, Android sensorOrientation. Confirm on-device via the HUD buffer WxH — a sideways face yields garbage crops.
- **Signing / OS gates.** iOS: signing identity, Developer Mode, "Always Allow", iOS 16.6 min. Android: API 24+. Camera permission prompt on first run.
- **Build config.** Use release on device (debug hangs on recent iOS/Xcode; the vendored xcframework is device-only, no simulator).
- **Non-determinism acceptance.** Server-side selection can change minute to minute. "It ran once" is not evidence — require clean runs across multiple cold starts and at least one fresh install; re-verify after any backend re-target.
- **Enrollment privacy.** Only the L2-normalized 128-d vector is stored (Keychain/Keystore); no face image is persisted.
- **Secrets.** The personal key is embedded in the client via gitignored lib/config/secrets.dart — never committed.

## Paste-back reconciliation status

- GATE-0 paste-back: **NOT yet arrived.** model_registry.dart holds placeholders `ajayshah/LiveGatePAD` v1 and `ajayshah/LiveGateFace` v1.
- Local ONNX contract (verified in onnxruntime, ground truth at export): PAD `input`[1,3,80,80]→`output`[1,3]; FACE `data`[1,3,112,112]→`fc1`[1,128]. Reconcile the dashboard's served shapes against these on paste-back — a mismatch is stop-the-line (rare; the local ONNX is almost always right).
- Injecting the registered name/version is one file, one commit. The device run requires it plus a real personal key in secrets.dart.

## References

- App directory: apps/LiveGate
- Core SDK: ZETIC Melange (zetic_mlange 1.8.1, Flutter FFI) — `ZeticMLangeModel.create(personalKey, name, version, modelMode, onProgress)`, `Tensor.float32List`, `run([...])`, `close()`
- Model 1 (PAD): garciafido/minifasnet-v2-anti-spoofing-onnx — MiniFASNet-V2, input `input` float32[1,3,80,80] BGR [0,255] (NOT ÷255), output `output` float32[1,3], LIVE = softmax[1] ≥ 0.45; Melange `ajayshah/LiveGatePAD`
- Model 2 (FACE): opencv/face_recognition_sface — SFace ArcFace, input `data` float32[1,3,112,112] BGR [0,255] (norm in-graph), output `fc1` float32[1,128] (L2-norm in Dart), MATCH if cosine ≥ 0.363; Melange `ajayshah/LiveGateFace`
- Face detect / landmarks: google_mlkit_face_detection (on-device, bundled, offline) — bbox + 5-pt (eyes/nose/mouth)
- Frameworks: Flutter, camera, google_mlkit_face_detection, flutter_secure_storage, flutter_launcher_icons; CoreML / Apple Neural Engine + TFLite (via Melange)
- Reference app: apps/FireDetectionYOLO/Flutter (PyroGuard) for structure + Android/iOS release config; LESSONS.md for the Android release gauntlet
- Bundle id: com.zeticai.livegate · Product name: TrueFace
- Test device: TBD (human, GATE 3)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
