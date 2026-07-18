# HANDOFF — ContentModeration (SafeLens)

> Status: GATE 3 — READY FOR DEVICE (not "done"). Tier A green, Tier B logged, Tier C surfaced. Model name/version are late-binding placeholders pending the GATE-0 paste-back; injecting them is a one-file change (lib/services/model_registry.dart) and is required before the device run.

## Goal

An on-device content-safety gate for Flutter (iOS + Android), powered by a ViT-Tiny NSFW/SFW classifier (Marqo/nsfw-image-detection-384) through the ZETIC Melange SDK. The user picks an image from the gallery or camera; the app preprocesses and classifies it entirely on-device (no cloud call), applies a softmax to the raw `[NSFW, SFW]` logits, and drives a KEEP / REVIEW-BLUR / BLOCK decision from score bands on P(NSFW) (0.30 / 0.70). It shows the picked image (blurred for REVIEW/BLOCK — the demo story), the decision with band color, P(NSFW)/P(SFW) meters, and an on-screen latency + diagnostics HUD. Product name: SafeLens.

## Todo List

- [x] Stage 0: export Marqo/nsfw-image-detection-384 to ONNX (opset 12, static [1,3,384,384], nsfw-vit-tiny-384.onnx) + sample_input.npy.
- [x] Stage 0: model_selection.md, melange_upload.md, demo_images/ (KEEP/REVIEW/BLOCK triplet + results.json) present.
- [x] Author HANDOFF.md and present the GATE-2 build plan + Tier A test list (approved).
- [x] Scaffold Flutter app under apps/ContentModeration/Flutter/ (org com.zetic, bundle id com.zetic.contentmoderation).
- [x] lib/services/model_registry.dart — SOLE owner of Melange name/version placeholders (`ajayshah/ContentModeration`, v1); verified nothing else references them.
- [x] lib/config/secrets.example.dart (tracked template) + lib/config/secrets.dart gitignored (real key never committed).
- [x] lib/services/preprocessor.dart — decode RGB, shortest-edge-384 BICUBIC+ANTIALIAS resampler (PIL coefficient formula, crop-aware fused single-pass resize->normalize->NCHW), floor dim + floor center-crop.
- [x] lib/services/postprocessor.dart — softmax over [NSFW(idx0), SFW(idx1)], P(NSFW)=softmax[0], KEEP/REVIEW-BLUR/BLOCK bands at 0.30/0.70.
- [x] lib/models/moderation_result.dart — {pNsfw, pSfw, decision, bandColor, logits}.
- [x] lib/services/melange_service.dart — create/warm-up/run/close via ModelRegistry constants + Tensor.float32List / asFloat32List.
- [x] lib/screens/loading_screen.dart — model download + warm-up with progress bar.
- [x] lib/screens/main_screen.dart — gallery + camera pick, image panel with blur-preview, decision banner, P meters, latency/diagnostics HUD.
- [x] lib/widgets + lib/theme.dart — verdict banner, score meter, diagnostics HUD, on-device badge.
- [x] Tier A tests: postprocessor_test, preprocessor_test, resize_fidelity_test (vs PIL/numpy golden + antialias guard), demo_pipeline_test — 38 tests green.
- [x] test/benchmark/hot_path_benchmark.dart — A4 mock-JPEG preprocess + softmax median.
- [x] test/fixtures/gen_golden.py — golden generator; reproduces results.json KEEP/BLOCK anchors + all 3 decisions via onnxruntime (stop-the-line check passed).
- [x] Tier B: applied crop-aware fused resampler (9.4x measured), warm-up, off-isolate decode; rejected flatten-coeffs + pre-resize with measured/justified reasons.
- [x] Custom SafeLens launcher icon (1024x1024 source -> flutter_launcher_icons, iOS AppIcon.appiconset + Android mipmaps).
- [x] Product name SafeLens: iOS CFBundleDisplayName, Android android:label, MaterialApp title, loading/app-bar text.
- [x] Android release hardening: INTERNET + ACCESS_NETWORK_STATE in MAIN manifest, AGP 8.9.1 / Kotlin 2.1.0 / Gradle 8.11.1, minify off, proguard keeps, useLegacyPackaging (LESSONS.md).
- [x] Tier A green: flutter analyze clean; `flutter build ios --release --no-codesign` (36.4MB) + `flutter build apk --release` (198.9MB) both pass with zero model bytes.
- [ ] **[BLOCKED – human (GATE 0 dashboard)]** Melange registered name/version paste-back not yet received; reconcile served I/O shapes vs spec, then inject into model_registry.dart (one file, one commit). Root cause: dashboard upload is human-only.
- [ ] **[BLOCKED – human (device run)]** Physical-device run pending: requires the name/version injection + real personal key. Read served target+apType from native console; confirm no MPSGraph GPU crash; verify latency across cold starts + one fresh install.

## Deliverables

- Flutter source under apps/ContentModeration/Flutter/ (main, loading + main screens, model_registry, MelangeService, preprocessor with bicubic-antialias resampler, postprocessor, moderation_result, theme + 4 widgets).
- Tier A test suite (test/postprocessor_test.dart, preprocessor_test.dart, resize_fidelity_test.dart, demo_pipeline_test.dart) + A4 hot-path benchmark; golden fixtures under test/fixtures/ + gen_golden.py generator.
- Custom SafeLens launcher icon (iOS + Android) from assets/icon/app_icon.png (reproducible via assets/icon/gen_icon.py); SafeLens display name across iOS/Android/in-app.
- Model assets (Stage 0, committed): export.py, nsfw-vit-tiny-384.onnx, sample_input.npy; registered Melange model ajayshah/ContentModeration (v1, pending GATE-0 confirmation).
- Platform config: iOS 16.6 (Podfile + pbxproj), NSPhotoLibraryUsageDescription + NSCameraUsageDescription; Android release gauntlet fixes (LESSONS.md).

## Validation report (Tier A)

- **A1 static analysis:** `flutter analyze` -> No issues found (0 errors, 0 warnings). No TODOs / stubs in shipped files.
- **A2 build:** iOS `flutter build ios --release --no-codesign` -> Built Runner.app (36.4MB). Android `flutter build apk --release` -> app-release.apk (198.9MB). Both build with ZERO model bytes (Melange downloads at runtime). Launcher icon + SafeLens name present. (AGP/Kotlin/Gradle "will soon be dropped" warnings are expected — the versions are intentionally pinned per LESSONS.md and the builds succeed.)
- **A3 unit tests (38, all green):** softmax correctness / applied-exactly-once / not-double-applied / not-a-sigmoid / numerically stable / rejects wrong length; label order idx0=NSFW,idx1=SFW (reversed) drives BLOCK vs KEEP; decision bands just-below/exactly/just-above 0.30 and 0.70 (inclusive-lower); normalization exactness ((v-0.5)/0.5, constant-127 image -> ~0, not plain /255, not ImageNet); RGB (not BGR) channel order; cubic kernel shape (Keys a=-0.5, negative anti-alias lobe); resize shortest-edge-384 + floor center-crop geometry; resize fidelity vs golden (see below); antialias-engaged guard; demo-image end-to-end.
- **Resize fidelity (the headline trap):** the Dart bicubic-antialias pipeline reproduces the numpy/PIL-formula golden tensors on lossless PNG fixtures to **maxAbs ~1e-6, meanAbs ~1e-7** (exact, only double-vs-float32 accumulation order differs). On the real demo JPEGs the delta is meanAbs ~0.007 (Dart-vs-libjpeg decode variance only — the resampler itself is proven exact on PNG). The antialias guard shows cubic-antialias differs from a plain bilinear resize by meanAbs 0.61 (a silent bilinear fallback would read ~0).
- **A4 hot-path micro-benchmark:** median **177 ms** for a 1440x1080 mock JPEG (JIT test VM; AOT device will be faster). Split: JPEG decode ~155 ms (in the `image` package), the pure-Dart resampler ~21 ms, softmax+banding negligible. This is the post-processing budget, NOT end-to-end device latency (NPU/CPU inference is fixed by Melange and only appears on hardware).
- **Reference reproduction (GATE-2 stop-the-line check):** the golden generator, via onnxruntime on the exact pipeline, reproduces results.json to |dP(NSFW)|: KEEP 0.0001, BLOCK 0.0108, REVIEW 0.0836. All three DECISIONS reproduce correctly. The REVIEW (classical-art) residual is the documented high-frequency resize-sensitivity case (model_selection.md: max Δ 0.22; even the canonical timm transform gives 0.21 here — our pipeline is closer to results.json than timm). Not a bug; reported transparently.

## Tier B optimization log

The rule: an applied optimization must show a measured >=0.5% improvement, or it is removed. Measured on the A4 hot path (resample portion, since JPEG decode is library-fixed).

- **APPLIED — crop-aware fused hand-written resampler:** 19.1 ms vs 179.8 ms for a naive `image`-package `copyResize` (whole frame) + crop + normalize path = **9.4x faster** (measured, warmed, 40 iters). Only the 384 rows/cols that survive the center crop are resampled; resize + /255 + (x-0.5)/0.5 + HWC->NCHW are fused into a single write with no intermediate full-resolution buffer.
- **APPLIED — preallocated typed buffers + typed-data views:** one `Float32List` per pass, `Uint8List`/`Float32List` views in the hot loops, no boxed `List` or per-pixel object allocation; loop-invariant hoisting (`k.length`, row/col base offsets).
- **APPLIED — pass order:** horizontal-first keeps the smaller source dimension in the intermediate for landscape photos (the common phone-camera case).
- **APPLIED — model lifecycle:** one warm-up dummy inference right after load (first real pick is not the cold one); decode+resample pushed off the UI isolate via `compute()`; a `_busy` guard prevents overlapping picks piling up.
- **REJECTED — flattened coefficient arrays:** measured ~23.6 ms vs the List<Float32List> form's ~21.3 ms (~10% SLOWER in the AOT-less test VM) and added complexity -> reverted per the 0.5% rule.
- **REJECTED — image_picker maxWidth/maxHeight pre-resize:** would offload the downscale to the platform's own (non-bicubic-antialias) resampler and break the fidelity the whole pipeline exists to preserve.
- **N/A — per-frame isolate reuse / frame-drop throttle / repaint-on-change:** no live-camera stream (single-shot pick, streaming meter descoped at GATE 2); the `_busy` guard covers the single-shot concurrency case.
- **NOTE:** JPEG decode (~155 ms) dominates the hot path and lives in the `image` package; it is not reducible without bypassing the exact transform, so it is accepted as-is.

## Runtime Risk (Tier C) — surfaced for the human device run

- **Served artifact:** this is a ViT; the client cannot force the backend. Expected realistic non-crashing fallback is CPU (TFLITE_FP16, ~hundreds of ms); NPU/Neural-Engine only if ZETIC serves a CoreML NE artifact. Read the ACTUAL served `target`+`apType` from the native console — that, not the requested mode, is ground truth.
- **modelMode:** RUN_AUTO. Do NOT expect any client mode to steer off a crashing artifact. The iOS/macOS 26.3+ MPSGraph GPU crash (attention-fusion pattern — the same class PyroGuard hit) is handled server-side by ZETIC filtering the GPU candidate for affected OS. If a new OS crashes at first inference, escalate to ZETIC to filter GPU for that OS.
- **Native observability:** `xcrun devicectl device process launch --console --terminate-existing --device <UDID> com.zetic.contentmoderation`. Dart `print`/`debugPrint` do NOT surface in a release device console, so all diagnostics (preprocess/inference ms, decoded buffer WxH, raw logits, P values) are on the in-app HUD by design.
- **Signing / OS gates (manual):** iOS signing identity/team, Developer Mode, "Always Allow", min iOS 16.6; Android minSdk 24. iOS simulator is a dead end (device-only ios-arm64 xcframework) — physical device only.
- **Build config:** run RELEASE on device (debug hangs on recent iOS/Xcode; a debug icon tap shows the "launch from Flutter tooling" screen — expected). Android release fixes are in place (INTERNET in MAIN manifest, R8 off + Melange keeps, useLegacyPackaging).
- **Network / cold start:** the model (~22 MB) downloads from S3 on first launch — a spinner on poor conference Wi-Fi. Pre-download / pre-warm and rehearse a fresh install. Requires INTERNET (fixed).
- **Non-determinism acceptance:** server-side selection can return a different artifact minute-to-minute. "It ran once" is not evidence — acceptance is clean runs across multiple cold starts and at least one fresh install, re-verified after any backend/model re-target.
- **Secrets:** the personal key is embedded in the client at build time (lib/config/secrets.dart, gitignored). Never commit it.

## Paste-back reconciliation status

- GATE-0 paste-back: **NOT yet received.** lib/services/model_registry.dart holds placeholders `ajayshah/ContentModeration` / version 1.
- Spec-expected served shapes to reconcile: input `float32[1,3,384,384]` NCHW RGB; output `float32[1,2]` raw logits [NSFW, SFW]. On a clean reconcile, inject the registered name/version into model_registry.dart (one file, one commit). A shape mismatch is stop-the-line.

## References

- App directory: apps/ContentModeration
- Core SDK: ZETIC Melange (zetic_mlange ^1.8.1, resolved 1.9.1 — identical `create` surface verified in pub-cache). Call: `ZeticMLangeModel.create(personalKey: zeticPersonalKey, name: ModelRegistry.modelName, version: ModelRegistry.modelVersion, modelMode: ModelMode.runAuto, onProgress: ...)` -> `Tensor.float32List(data, shape: [1,3,384,384])` -> `model.run([input])` -> `outputs.first.asFloat32List()` -> `model.close()`.
- Model: Marqo/nsfw-image-detection-384 (timm vit_tiny_patch16_384, 22 MB). Input float32[1,3,384,384] NCHW RGB, normalized (x-0.5)/0.5 -> [-1,1]. Output float32[1,2] RAW LOGITS, order [NSFW(idx0), SFW(idx1)].
- Preprocessing (must match timm eval): shortest-edge-384 BICUBIC+antialias -> center-crop 384 -> /255 -> (x-0.5)/0.5 -> NCHW RGB.
- Decision bands: KEEP P(NSFW)<0.30 · REVIEW/BLUR 0.30<=P(NSFW)<0.70 · BLOCK P(NSFW)>=0.70.
- Reference values: demo_images/results.json (KEEP 0.056 / REVIEW 0.489 / BLOCK 0.754); model_selection.md (100% SFW specificity, max FP on safe 0.081).
- Spec: apps/ContentModeration/SPEC_STUB.md (approved GATE-1). Docs: CLAUDE.md, AGENTS.md, VALIDATION.md, LESSONS.md.
- Reference apps studied: RetinaDRScreen (FundusGate — closest: binary softmax classifier, image_picker), FireDetectionYOLO (PyroGuard — Melange lifecycle, Android release fixes), ShelfScanYOLO.
- Test device: TBD by human at GATE 3 (iOS 16.6+ / Android minSdk 24).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
