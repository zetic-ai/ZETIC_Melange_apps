# HANDOFF — VisualSearch (SnapSeek)

> Status: GATE 3 — READY FOR DEVICE. Full app built, Tier A green (34 tests), Tier B logged, Tier C filled. Melange name/version paste-back is still BLOCKED at GATE 0 (does not block the build); the constants file holds proposed placeholders and needs a one-line injection before the device run.

## Goal

A fully on-device, two-model visual-search demo for Flutter (iOS + Android), powered by the ZETIC Melange SDK. The user snaps one photo of any product/object; a YOLO11n detector localizes the salient item, the crop is embedded by a MobileCLIP2-S0 image tower into a 512-d L2-normalized vector, and that vector is ranked by cosine (dot product) against a bundled demo gallery of precomputed catalog vectors to show the top-K visual matches. The story: remove the per-tap server-GPU embedding call and get an instant, offline result for fashion e-commerce (the "Musinsa wedge"). Only the vector would ever leave the device.

## Todo List

- [x] Stage-0 export recipe authored + re-run (export.py — both models; ONNX contracts locally verified: detect [1,3,640,640]→[1,84,8400] top score 0.853, embed [1,3,256,256]→[1,512] unit-norm, cosine 1.000000 vs torch).
- [x] Stage-0 model selection + upload instructions (model_selection.md, melange_upload.md, SPEC_STUB.md).
- [x] Scaffold Flutter app under Flutter/ (org com.zeticai, package snapseek, iOS+Android).
- [x] lib/services/model_registry.dart — the ONE late-binding file holding BOTH models' name/version placeholders; verified nothing else references them (only melange_service.dart imports ModelRegistry).
- [x] Dual-model MelangeService: two ZeticMLangeModel.create instances, warm-up both, full decode→detect→crop→embed→rank pipeline, close both.
- [x] Detector pre/post: letterbox 640 (pad 114) /255 RGB NCHW; decode [1,84,8400] channel-major; threshold 0.25; cxcywh→xyxy; un-letterbox; GLOBAL (class-agnostic) NMS; primary = top-conf box; no-detection center-crop fallback.
- [x] Embed pre/post: crop primary box (+6% margin) from original-frame px → bilinear resize 256×256 → /255 (NO mean/std) → NCHW; output already unit-norm, cosine = dot product.
- [x] Bundled demo gallery: 60 real deepfashion product images embedded through the REAL regenerated visualsearch_embed.onnx; vectors (gallery.json, 320 KB) + thumbnails committed; ONNX stays gitignored. Self-check margin 0.339 (same-instance 0.818 vs cross 0.478).
- [x] Capture + UI: live viewfinder, snap button, primary box overlay (teal) + embedded-crop rect (amber dashed), top-K match strip with cosine + SAME-PRODUCT badge, on-HUD diagnostics (detect ms / embed ms / total ms, frame WxH, primary box coords).
- [x] Tier A tests (9 trap files + helpers + real-gallery asset test = 34 tests) — all green; flutter analyze clean (0 issues).
- [x] Tier A4 + Tier B: hot-path micro-benchmark (mock tensors, full Dart path) with before/after deltas per optimization (0.5% rule).
- [x] Custom launcher icon (SnapSeek aperture+magnifier glyph, 1024×1024) via flutter_launcher_icons (iOS + Android, remove_alpha_ios).
- [x] Product name SnapSeek as display name (iOS CFBundleDisplayName, Android android:label, in-app title); bundle id / folder / Melange model name unchanged.
- [x] Android release gauntlet (LESSONS.md): AGP 8.9.1 / Kotlin 2.1.0 / Gradle 8.11.1, minify+shrink off, legacy jniLibs packaging, INTERNET + ACCESS_NETWORK_STATE + CAMERA in the MAIN manifest, suppressUnsupportedCompileSdk. Release APK built (202 MB).
- [x] iOS release build: NSCameraUsageDescription, iOS 16.6 min; flutter build ios --release --no-codesign built Runner.app (37.7 MB).
- [ ] **[BLOCKED – human/GATE-0]** Inject registered Melange name/version for BOTH models into model_registry.dart once the dashboard paste-back arrives (expected ajayshah/VisualSearchDetect v1, ajayshah/VisualSearchEmbed v1). Currently proposed placeholders; build proceeds against them.
- [ ] **[BLOCKED – human]** Physical-device run (iOS/Android release) + inject the personal key via adapt_mlange_key.sh; read served target+apType from the native console; confirm no first-inference MPSGraph crash on the FastViT embed tower (iOS 26.3+).

## Deliverables

- Flutter source under apps/VisualSearch/Flutter/ (main, theme, loading + main screens, dual-model MelangeService, preprocessor, detector postprocessor, global NMS, gallery ranking, models, overlay/HUD/match-strip widgets).
- Late-binding constants: lib/services/model_registry.dart (both models' name/version placeholders — the only file referencing them).
- Bundled demo gallery assets: assets/gallery/gallery.json (60×512 vectors from the real exported ONNX) + assets/gallery/thumbs/*.jpg.
- Tier A test suite (test/*.dart + test_helpers.dart) + test/benchmark/hot_path_benchmark.dart with Tier B before/after deltas.
- Custom launcher icon (assets/icon/app_icon.png) + SnapSeek product name.
- Android + iOS release build configuration carrying all LESSONS.md fixes.
- This HANDOFF.md (finalized).

## Validation report

### Tier A results (all green, no device)

- A1 static analysis: `flutter analyze` → No issues found (0 errors, 0 warnings, 0 info).
- A2 build: iOS `flutter build ios --release --no-codesign` → Runner.app 37.7 MB; Android `flutter build apk --release` → app-release.apk 202 MB (large by design: R8 minify/shrink disabled per LESSONS.md Bug 1). Custom launcher icon + SnapSeek display name present on both platforms. Both build with ZERO model bytes (Melange downloads at runtime).
- A3 unit tests: 34 tests across 10 files, all passing. Trap coverage: channel-major [1,84,8400] decode (anchor-stride + high-index anchor + wrong-length assert), letterbox inverse round-trip (4 source sizes + geometry constants), crop-space mapping (original-frame px + margin clamp + wrong-region guard), global class-agnostic NMS (overlapping different-class → one; contrasts per-class), threshold boundary (just-below/at/above 0.25), embed preprocessing (plain /255, NO mean/std; red→R=1 channel order), no-detection center-crop fallback, orientation (wide box stays wide, no transpose), gallery dot-product ranking (cosine(v,v)=1, descending, top-K), real bundled gallery unit-norm + self-retrieval.
- A4 hot-path micro-benchmark (median of 40, mock 1080×1920 frame + [1,84,8400] + 60×512 gallery, JIT test mode — a budget, not device latency): full Dart hot path ≈ 3.6–3.9 ms.

### Tier B optimization log (0.5% rule — measured on A4)

- Fused single-pass letterbox + pre-allocated Float32List (resize+/255+NCHW in one pass) vs naive 2-pass with an intermediate buffer: ~2.0 ms vs ~3.5 ms → ~40% faster. KEPT.
- Threshold-before-geometry in the decoder (skip cxcywh→xyxy + un-letterbox + Detection allocation for the ~8380/8400 sub-threshold anchors) vs geometry+alloc for every anchor: ~0.71 ms vs ~0.84 ms → ~15% faster. KEPT.
- Fused bilinear crop→256 resize straight into planar NCHW (no intermediate crop image): ~0.72 ms.
- Gallery ranking as a plain dot product over unit vectors (no per-item normalization, embed is unit-norm in-graph): ~0.03 ms for 60×512.
- Pre-allocated warm-up inference for BOTH models after load so the first real snap is not the slow one. KEPT (device-only effect; no A4 delta).
- Off-thread heavy Dart via compute() (decode+letterbox, decode+NMS, crop+resize) so the UI stays responsive during a snap; both native model.run calls stay on the main isolate (SDK binds each handle to its creating isolate); native output copied before reuse / before crossing the isolate boundary. KEPT.
- Skipped (justified): per-frame throttling/_busy for a stream — N/A, this is snap-triggered single-shot with a _busy guard on the snap button; repaint-on-change for a live overlay — N/A, the results overlay paints once per snap.

## Runtime Risk (Tier C) — device-only, surfaced not tested

- Served artifact: expect CPU (TFLITE_FP16, hundreds of ms) as the realistic non-crash fallback, NOT NPU — matches PyroGuard. Client cannot force the backend. Read the ACTUAL served target+apType from the native console (`runtimeApType=...`), not the dashboard row. Two artifacts to watch (detect + embed) — check both.
- modelMode: RUN_AUTO for both models. Do NOT expect any modelMode to steer off a crashing artifact.
- Known crash path: the EMBED tower is a FastViT hybrid with MHSA self-attention — the exact ViT-style attention family that hit the iOS/macOS 26.3+ CoreML-GPU MPSGraph crash. If first inference aborts (SIGABRT, "MLIR pass manager failed"), it is not client-fixable; escalate to ZETIC to filter the GPU candidate server-side for that OS. The detector (all-conv YOLO11n) is lower risk.
- Native observability: `xcrun devicectl device process launch --console --terminate-existing --device <UDID> com.zeticai.snapseek` (iOS); `adb logcat -b crash -d` + `adb shell dumpsys package com.zeticai.snapseek | grep INTERNET` (Android). Dart print does NOT reach the console in release — all diagnostics (detect ms / embed ms / total, frame WxH, primary box coords) are on the in-app HUD.
- Signing + OS gates: iOS needs a signing identity (build was --no-codesign), Developer Mode, "Trust" the profile, iOS 16.6+. Android release is debug-signed for the demo.
- Build config: use RELEASE on device (debug hangs on recent iOS/Xcode; the vendored xcframework is device-only — no simulator slice).
- Network / cold start: both models download on first launch (~10.7 MB + ~45.8 MB) over the network; on poor conference Wi-Fi that is a spinner on the loading screen. Pre-download / pre-warm on good Wi-Fi and rehearse a fresh install. Watch for the large-model checksum-mismatch issue (LESSONS.md) on the 45.8 MB embed model; if it recurs on a clean re-download, escalate to ZETIC.
- Non-determinism acceptance: server-side selection can return a different artifact minute to minute. "It ran once" is not evidence. Acceptance = runs cleanly across multiple cold starts and at least one fresh install, re-verified after any backend/model re-target — and confirm BOTH models load and infer.
- Secrets: the personal key is embedded in the client. Inject via adapt_mlange_key.sh (replaces `const _sourceMlangeKey = "YOUR_MLANGE_KEY"`); never commit a real key.

## Dual-model lifecycle notes (first two-model app — reuse this pattern)

- Load order: create the DETECT model first (10.7 MB), then EMBED (45.8 MB), then load the bundled gallery, then warm up. Sequential awaits — the SDK's create() is async and each download is independent, but sequential keeps the loading-bar progress monotonic and avoids two large concurrent downloads competing on conference Wi-Fi. Progress is split 0–45% detect, 45–90% embed, 90–100% gallery+warm-up.
- Warm-up: one dummy zero-tensor inference per model immediately after both load (kDetectSize and kEmbedSize zeros). This pays the one-time graph-compile/first-inference cost off the critical path so the first real snap is fast. Do both, not just the first model.
- Isolate binding: each ZeticMLangeModel handle is bound to the isolate that created it, so BOTH model.run() calls must happen on the same (main) isolate that ran init(). Only the pure-Dart pre/post work goes to compute() isolates. Do not try to run one model in a background isolate.
- Buffer copies: asFloat32List() returns a view over a reused native buffer — copy (Float32List.fromList) the detector output before the embed run and before any compute() hop; likewise copy the embedding before ranking. With two models sharing the runtime this matters more, not less.
- Memory: both models are resident simultaneously (~56 MB of model weights + two native runtime contexts). Acceptable on a modern phone; it is the reason to embed a CROP (256×256) rather than keep large intermediates around. The decoded working frame is capped at a 1280 long side to bound the cross-isolate copy and the crop cost.
- Teardown: close() BOTH models in dispose() (guard on isClosed); a single-model teardown would leak the second native context.
- Late-binding: both models' name+version live in ONE model_registry.dart with clearly-marked placeholders; melange_service.dart is the only referencer. A two-model paste-back is still a one-file edit.

## License flags (surface at GATE 3 / PR — accepted for the internal demo)

- EMBED weights (MobileCLIP2-S0, timm fastvit_mci0.apple_mclip2_dfndr2b): `apple-amlr` (Apple ML Research) — non-standard, potentially research-restricted. Fine for an on-device trade-show demo (only vectors would leave the device); MUST be legally reviewed before any commercial productization. Permissive fallbacks (weaker): TinyCLIP-8M (MIT), DINOv2-small (Apache-2.0).
- DETECT weights (Ultralytics YOLO11n): AGPL-3.0 (copyleft) — established repo precedent (6+ shipped YOLO apps); honor/review before commercial productization.
- Bundled gallery images (Marqo/deepfashion-inshop, 60 photos): research-use-only license — used ONLY to precompute demo catalog vectors + thumbnails for the internal demo; not for redistribution/commercial use. Swap for clearly-licensed product photos before any external release.

## Paste-back reconciliation status

- GATE-0 paste-back: NOT yet arrived. model_registry.dart holds PROPOSED placeholders (ajayshah/VisualSearchDetect v1, ajayshah/VisualSearchEmbed v1) clearly marked `[LATE-BINDING — placeholder until GATE-0 paste-back]`.
- Local ONNX contract (ground truth at export): detect images[1,3,640,640] → output0[1,84,8400]; embed image[1,3,256,256] → embedding[1,512] unit-norm. Reconcile the served shapes against these when the paste-back arrives; a mismatch is stop-the-line (rare — the local ONNX is verified).
- Injection is a one-file, one-commit change in model_registry.dart; the device run requires it plus the personal-key injection.

## References

- App directory: apps/VisualSearch
- Core SDK: ZETIC Melange (zetic_mlange 1.9.1, Flutter FFI) — dual-model. Bundle id com.zeticai.snapseek.
- Model 1 (detector): Ultralytics YOLO11n COCO — input float32[1,3,640,640] RGB letterboxed /255, output float32[1,84,8400] channel-major, NMS not baked in, 80 COCO classes (used as salient-object localizer)
- Model 2 (embedding): MobileCLIP2-S0 image tower (timm fastvit_mci0.apple_mclip2_dfndr2b) — input float32[1,3,256,256] RGB /255 no mean/std, output float32[1,512] L2-normalized in-graph
- Gallery source: Marqo/deepfashion-inshop (60 real product photos, 16 instances × 3-4 views)
- Reference apps studied: FireDetectionYOLO (PyroGuard) + SafetyPPEYOLO (SiteGuard); no built dual-model app exists in-repo (LiveDocRedact/SignTranslate/VoxScribe are empty stubs) — dual lifecycle designed here (see notes above)
- Capture deviation: SnapSeek uses takePicture() still-JPEG + EXIF bakeOrientation (approved), NOT PyroGuard's live YUV/BGRA image-stream — right call for snap-then-search, and it sidesteps per-frame plane wrangling / most orientation traps
- Test device: TBD (human, GATE 3)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
