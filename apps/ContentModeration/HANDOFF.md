# HANDOFF — ContentModeration (SafeLens)

> Status: GATE 2 (approach) — plan-of-record before writing app code. Build runs in parallel with the GATE-0 Melange upload; model name/version are late-binding placeholders in one file until the paste-back.

## Goal

An on-device content-safety gate for Flutter (iOS + Android), powered by a ViT-Tiny NSFW/SFW classifier (Marqo/nsfw-image-detection-384) through the ZETIC Melange SDK. The user picks an image from the gallery or camera; the app preprocesses and classifies it entirely on-device (no cloud call), applies a softmax to the raw `[NSFW, SFW]` logits, and drives a KEEP / REVIEW-BLUR / BLOCK decision from score bands on P(NSFW) (0.30 / 0.70). Shows the picked image (with a blur-preview for REVIEW/BLOCK), the decision with band color, P(NSFW)/P(SFW) meters, and an inference-latency HUD. Product name: SafeLens.

## Todo List

- [x] Stage 0: export Marqo/nsfw-image-detection-384 to ONNX (opset 12, static [1,3,384,384], nsfw-vit-tiny-384.onnx) + sample_input.npy.
- [x] Stage 0: model_selection.md, melange_upload.md, demo_images/ (KEEP/REVIEW/BLOCK triplet + results.json) present.
- [x] Author this HANDOFF.md and present the GATE-2 build plan + Tier A test list.
- [ ] Scaffold Flutter app under apps/ContentModeration/Flutter/ (flutter create, then repo anatomy).
- [ ] lib/services/model_registry.dart — SOLE owner of Melange name/version placeholders (`ajayshah/ContentModeration`, v1); nothing else references them.
- [ ] lib/config/secrets.example.dart (template) + gitignore lib/config/secrets.dart (real key, never committed).
- [ ] lib/services/preprocessor.dart — decode RGB, shortest-edge-384 BICUBIC+antialias resize (PIL-matching separable resampler), center-crop 384, /255, (x-0.5)/0.5, HWC->NCHW [1,3,384,384].
- [ ] lib/services/postprocessor.dart — softmax over [NSFW, SFW], P(NSFW)=softmax[0], KEEP/REVIEW-BLUR/BLOCK bands at 0.30/0.70.
- [ ] lib/models/moderation_result.dart — data class {pNsfw, pSfw, decision, bandColor, logits}.
- [ ] lib/services/melange_service.dart — create/warm-up/run/close via ModelRegistry constants + Tensor.float32List/asFloat32List.
- [ ] lib/screens/loading_screen.dart — model download + warm-up with progress bar.
- [ ] lib/screens/main_screen.dart — gallery + camera pick, image panel with blur-preview, decision banner, P meters, latency HUD.
- [ ] lib/widgets + lib/theme.dart — decision banner, score meters, diagnostics HUD.
- [ ] test/ Tier A: postprocessor_test, preprocessor_test, resize_fidelity_test (vs PIL golden), demo_pipeline_test.
- [ ] test/benchmark/hot_path_benchmark.dart — mock-image preprocess + softmax median (A4 baseline).
- [ ] Tier B: apply optimization checklist levers with measured before/after deltas on the A4 benchmark.
- [ ] Custom launcher icon (1024x1024 source -> flutter_launcher_icons, iOS + Android, remove_alpha_ios).
- [ ] Product name SafeLens: iOS CFBundleDisplayName, Android android:label, MaterialApp title/app bar.
- [ ] Android release hardening: INTERNET + ACCESS_NETWORK_STATE in MAIN manifest, AGP 8.9.1 / Kotlin 2.1.0 / Gradle 8.11.1, minify off, proguard keeps, useLegacyPackaging (per LESSONS.md).
- [ ] Tier A green: flutter analyze clean, `flutter build ios --release --no-codesign` + Android release build pass with zero model bytes.
- [ ] **[BLOCKED – human (GATE 0 dashboard)]** Melange registered name/version paste-back, served I/O shape reconciliation, then one-file injection into model_registry.dart.
- [ ] **[BLOCKED – human (device run)]** Physical-device run: read served target+apType from native console; confirm no MPSGraph GPU crash; verify latency across cold starts + a fresh install.

## Deliverables

- Flutter source under apps/ContentModeration/Flutter/ (screens, model_registry, MelangeService, preprocessor with bicubic-antialias resampler, postprocessor, result model, widgets/theme).
- Tier A test suite + A4 hot-path micro-benchmark, all green without a device.
- Custom SafeLens launcher icon (iOS AppIcon.appiconset + Android mipmaps) and SafeLens display name.
- Model assets already committed: export.py, nsfw-vit-tiny-384.onnx, sample_input.npy; registered Melange model ajayshah/ContentModeration (v1, pending GATE-0 confirmation).
- This HANDOFF.md (plan-of-record, finalized at GATE 3 with validation report + Tier B log + Tier C checklist).

## References

- App directory: apps/ContentModeration
- Core SDK: ZETIC Melange (zetic_mlange 1.8.1, Flutter FFI); create(personalKey, name, version, modelMode: runAuto, onProgress) -> Tensor.float32List(shape:[1,3,384,384]) -> run -> outputs.first.asFloat32List() -> close.
- Model: Marqo/nsfw-image-detection-384 (timm vit_tiny_patch16_384, 22 MB). Input float32[1,3,384,384] NCHW RGB, normalized (x-0.5)/0.5 -> [-1,1]. Output float32[1,2] RAW LOGITS, order [NSFW(idx0), SFW(idx1)].
- Preprocessing (must match timm eval): shortest-edge-384 BICUBIC+antialias -> center-crop 384 -> /255 -> (x-0.5)/0.5 -> NCHW RGB.
- Decision bands: KEEP P(NSFW)<0.30 · REVIEW/BLUR 0.30<=P(NSFW)<0.70 · BLOCK P(NSFW)>=0.70.
- Reference values: demo_images/results.json (KEEP 0.056 / REVIEW 0.489 / BLOCK 0.754), model_selection.md (100% SFW specificity, max FP on safe 0.081).
- Spec: apps/ContentModeration/SPEC_STUB.md (approved GATE-1). Docs: CLAUDE.md, AGENTS.md, VALIDATION.md, LESSONS.md.
- Reference apps studied: RetinaDRScreen (FundusGate — closest: binary softmax classifier, image_picker, (x-0.5)/0.5), FireDetectionYOLO (PyroGuard — Melange lifecycle, Android release fixes), ShelfScanYOLO.
- Test device: TBD by human at GATE 3 (iOS 16.6+ / Android minSdk 24).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
