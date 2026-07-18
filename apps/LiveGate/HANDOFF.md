# HANDOFF — LiveGate (TrueFace)

> Status: **GATE 2 (approach approval)** — plan-of-record before any app code. Two Melange models (PAD + FACE) are late-binding placeholders pending the GATE-0 paste-back. Build runs in parallel with the upload.

## Goal

A fully on-device KYC live-selfie gate for Flutter (iOS + Android), powered by two ZETIC Melange models: a MiniFASNet-V2 anti-spoof classifier (PAD) and an SFace ArcFace face-embedding model. Point the front camera at a face; the app detects the face app-side (bbox + 5 landmarks), decides **LIVE vs SPOOF** (softmax class 1 ≥ 0.45), and for a real face returns a **1:1 cosine match** against a locally-enrolled reference vector (≥ 0.363). A SPOOF fails the gate regardless of match; only LIVE + MATCH passes. No face bytes ever leave the device, and enrollment stores only the 128-d vector.

## Todo List

- [x] Stage 0: export + validate both ONNX (pad_minifasnet_v2.onnx [1,3,80,80]→[1,3]; face_sface.onnx `data`[1,3,112,112]→`fc1`[1,128]); demo goldens reproduced (PAD live=1.000, ÷255=0.006).
- [ ] Create Flutter app skeleton under apps/LiveGate/Flutter/ per CLAUDE.md §4 anatomy (from FireDetectionYOLO reference).
- [ ] `lib/services/model_registry.dart`: the ONLY file referencing Melange names/versions — both models as clearly-marked placeholders (name/version ONLY; no key).
- [ ] `lib/config/secrets.example.dart` (tracked, `<ZETIC_PERSONAL_KEY>` template) + gitignored `lib/config/secrets.dart` for the real key — RetinaDRScreen convention.
- [ ] `lib/services/melange_service.dart`: create/warm/run/close BOTH models (PAD + FACE) on the main isolate; dummy warm-up inference each.
- [ ] Face-detect stage: integrate google_mlkit_face_detection (on-device, bundled, offline) → bbox + 5 landmarks (eyes/nose/mouth). Repo MediaPipe folders (`apps/MediaPipe-Face-Detection`, `apps/MediaPipe-Face-Landmarker`) inspected and REJECTED as native-only demos (Kotlin/Swift, no Flutter/Dart to reuse; Landmarker is a 468-pt mesh) — reusing them would mean 4 Melange loads + porting SSD/mesh decode to Dart.
- [ ] `lib/services/preprocessor.dart`: BGR conversion; PAD 2.7× clip-shift crop → 80×80 BGR [0,255] NCHW; runs in compute() isolate.
- [ ] `lib/services/face_align.dart`: 5-point similarity/affine warp to ArcFace 112×112 template → BGR [0,255] NCHW.
- [ ] `lib/services/gate.dart`: PAD softmax + class-1 liveness threshold; SFace L2-norm + cosine; gate composition (SPOOF → reject, hide score).
- [ ] `lib/services/enrollment.dart`: enroll → store L2-normalized 128-d vector in flutter_secure_storage (Keychain/Keystore); no image persisted.
- [ ] UI: front-camera preview + face box; big LIVE/SPOOF verdict; match score + PASS/FAIL badge; enroll action; "on-device · no cloud" badge; per-stage latency + buffer WxH HUD (no Dart print on release).
- [ ] Tier A tests (10 tests — see `## Tier A Test Plan`), including a Python-generated golden preprocessing fixture.
- [ ] `test/benchmark/hot_path_benchmark.dart`: mock-tensor micro-benchmark of the pure-Dart hot path; record median baseline.
- [ ] Tier B optimization pass with measured deltas (≥0.5% rule).
- [ ] Custom launcher icon (face-in-gate/shield glyph) via flutter_launcher_icons (iOS + Android, remove_alpha_ios).
- [ ] Product name **TrueFace** as display name (iOS CFBundleDisplayName, Android android:label, in-app title); folder/bundle-id/Melange names unchanged.
- [ ] Android release gauntlet per LESSONS.md (AGP 8.9.1 / Kotlin 2.1.0 / Gradle 8.11.1, minify off, INTERNET + ACCESS_NETWORK_STATE, proguard keeps, legacy jniLibs packaging).
- [ ] iOS release build config (iOS 16.6, NSCameraUsageDescription, `flutter build ios --release --no-codesign`).
- [ ] `flutter analyze` clean; Tier A green; fill Tier C.
- [ ] **[BLOCKED – human/ZETIC dashboard]** GATE-0 paste-back: registered names/versions + served shapes for `ajayshah/LiveGatePAD` and `ajayshah/LiveGateFace`. Until then model_registry.dart holds placeholders. Root cause: model registration is the one manual dashboard step; injection is a one-file, one-commit change.

## Deliverables

- Flutter source under apps/LiveGate/Flutter/ (two-model MelangeService, ML Kit face detect, preprocessor, face_align, gate, enrollment, screens, widgets/HUD).
- Model assets (Stage 0, committed): export.py, pad_minifasnet_v2.onnx, face_sface.onnx, sample_input_pad.npy, sample_input_face.npy.
- lib/services/model_registry.dart — single late-binding file for both model name/version placeholders (no key); lib/config/secrets.example.dart template + gitignored secrets.dart for the key.
- Tier A test suite + Python-generated golden fixture; hot-path micro-benchmark.
- Custom launcher icon + TrueFace product name.
- This HANDOFF.md (finalized at GATE 3 with Tier A results, Tier B log, Tier C checklist, paste-back reconciliation).

## Build Plan

- **Pipeline (per frame, `_busy`-guarded):** copy camera planes → FrameData; run ML Kit face detection (native, async) on the frame → best bbox + 5 landmarks; if no face, HUD "no face" and release. In a compute() isolate build the PAD tensor (2.7× clip-shift crop → 80×80 BGR [0,255] NCHW). On the main isolate run PAD → softmax → liveScore. **If LIVE**, build the FACE tensor (5-pt warp → 112×112 BGR [0,255] NCHW) and run FACE → L2-norm → cosine vs enrolled (skipping FACE on SPOOF saves the 38 MB model run and matches the gate). Compose verdict; setState HUD.
- **Threading:** heavy pure-Dart (BGR convert, crop, warp) in compute() isolates (per FireDetectionYOLO precedent); both `model.run` calls on the main isolate (handle bound to it); ML Kit runs on its own native threads.
- **Late-binding:** model_registry.dart exposes `padModel` and `faceModel` as `(name, version)` placeholder records ONLY (no key). The personal key follows the repo secrets convention: tracked `lib/config/secrets.example.dart` template + gitignored `lib/config/secrets.dart`. melange_service.dart consumes both. Thresholds (0.45, 0.363) live in gate.dart, NOT in model_registry.

## Tier A Test Plan

- [ ] T1 PAD input range / ÷255-saturation guard: preprocessor keeps BGR [0,255] (max may exceed 1.0); a ÷255 tensor differs — the ÷255 path is the bug.
- [ ] T2 PAD live-class index: decoder reads softmax[1] as liveness (not [0]); hand-built logits verify.
- [ ] T3 PAD 2.7× crop clip-SHIFT round-trip: scale = min((H-1)/bh,(W-1)/bw,2.7); edge bbox shifts fully in-bounds (dims preserved), not one-edge-clamped.
- [ ] T4 FACE channel order + range: warp output is BGR, [0,255], no Dart normalization.
- [ ] T5 FACE L2-norm semantics: unit(v)·unit(v)=1.0; cosine∈[-1,1]; un-normalized dot ≠ cosine.
- [ ] T6 FACE 5-pt alignment: affine from src→template maps the 5 points within tolerance; template→template is identity.
- [ ] T7 Threshold boundaries: PAD 0.45 and FACE 0.363, just-below rejected / just-above accepted.
- [ ] T8 Gate composition: SPOOF + high cosine → REJECT (never PASS, score hidden); LIVE + high → PASS; LIVE + low → NO-MATCH.
- [ ] T9 Orientation transform round-trips a known box for the assumed buffer orientation.
- [ ] T10 Golden fidelity: Dart PAD preprocessor on a committed known input reproduces the Python-generated golden Float32List within tolerance (BGR + [0,255] + NCHW + resize match the ONNX contract).
- [ ] T11 (GATE-3 requirement) End-to-end separation sanity: through the actual 5-pt-aligned pipeline, aligned same-person cosine sits comfortably above 0.363 and different-person comfortably below (absolute RESULTS.md numbers not reproduced; proves the threshold still separates after alignment).

## Runtime Risk (Tier C)

- To be filled at GATE 3 (served artifact expectation, modelMode, device-console command, signing/build config, network/cold-start with FOUR-vs-TWO model loads, non-determinism acceptance).

## References

- App directory: apps/LiveGate
- Core SDK: ZETIC Melange (zetic_mlange 1.8.1, Flutter FFI) — `ZeticMLangeModel.create(personalKey, name, version, modelMode)`, `Tensor.float32List`, `run([...])`, `close()`
- Model 1 (PAD): garciafido/minifasnet-v2-anti-spoofing-onnx — MiniFASNet-V2, input `input` float32[1,3,80,80] BGR [0,255] (NOT ÷255), output `output` float32[1,3], LIVE = softmax[1] ≥ 0.45; Melange `ajayshah/LiveGatePAD`
- Model 2 (FACE): opencv/face_recognition_sface — SFace ArcFace, input `data` float32[1,3,112,112] BGR [0,255] (norm in-graph), output `fc1` float32[1,128] (L2-norm in Dart), MATCH if cosine ≥ 0.363; Melange `ajayshah/LiveGateFace`
- Face detect / landmarks: google_mlkit_face_detection (on-device, bundled, offline) — bbox + 5-pt (eyes/nose/mouth)
- Frameworks: Flutter, camera, google_mlkit_face_detection, flutter_secure_storage, flutter_launcher_icons; CoreML / Apple Neural Engine + TFLite (via Melange)
- Reference app: apps/FireDetectionYOLO/Flutter (PyroGuard) for structure + Android/iOS release config
- Test device: TBD (human, GATE 3)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
