# HANDOFF — VisualSearch (SnapSeek)

> Status: GATE 2 (approach) — plan-of-record submitted, awaiting orchestrator continue/redirect. No app code written yet. Melange name/version paste-back is BLOCKED at GATE 0 (does not block the build).

## Goal

A fully on-device, two-model visual-search demo for Flutter (iOS + Android), powered by the ZETIC Melange SDK. The user snaps one photo of any product/object; a YOLO11n detector localizes the salient item, the crop is embedded by a MobileCLIP2-S0 image tower into a 512-d L2-normalized vector, and that vector is ranked by cosine (dot product) against a bundled demo gallery of precomputed catalog vectors to show the top-K visual matches. The story: remove the per-tap server-GPU embedding call and get an instant, offline result for fashion e-commerce (the "Musinsa wedge"). Only the vector would ever leave the device.

## Todo List

- [x] Stage-0 export recipe authored (export.py — both models, re-runnable, ONNX contracts locally verified: detect [1,3,640,640]→[1,84,8400], embed [1,3,256,256]→[1,512] unit-norm, cosine 1.000000 vs torch).
- [x] Stage-0 model selection + upload instructions (model_selection.md, melange_upload.md, SPEC_STUB.md).
- [ ] Scaffold Flutter app under Flutter/ (flutter create, package name, folder anatomy per CLAUDE.md §4).
- [ ] lib/services/model_registry.dart — the ONE late-binding file holding BOTH models' name/version placeholders (VisualSearchDetect + VisualSearchEmbed); nothing else references them.
- [ ] Dual-model MelangeService: two ZeticMLangeModel.create instances, warm-up both, full detect→crop→embed→rank pipeline, close both.
- [ ] Detector pre/post: letterbox 640 (pad 114) /255 RGB NCHW; decode [1,84,8400] channel-major; threshold 0.25; cxcywh→xyxy; un-letterbox; GLOBAL (class-agnostic) NMS; primary = top-conf box; no-detection center-crop fallback.
- [ ] Embed pre/post: crop primary box (+5-8% margin) from original-frame px → resize 256×256 → /255 (NO mean/std) → NCHW; output is already unit-norm, cosine = dot product.
- [ ] Bundled demo gallery: regenerate visualsearch_embed.onnx via export.py, embed the 60 deepfashion product images through the REAL ONNX, bundle vectors (JSON) + thumbnails as Flutter assets; cosine top-K ranking.
- [ ] Capture + UI: live viewfinder, snap button, primary box overlay, embedded-crop preview, top-K match strip with cosine scores, on-HUD diagnostics (detect ms / embed ms, buffer WxH, primary box coords).
- [ ] Tier A tests (9 files + helpers) — see Validation section; all green via flutter analyze + flutter test.
- [ ] Tier A4 + Tier B: hot-path micro-benchmark (mock tensors, full Dart path) with before/after deltas per optimization (0.5% rule).
- [ ] Custom launcher icon (SnapSeek glyph, 1024×1024) via flutter_launcher_icons (iOS + Android, remove_alpha_ios).
- [ ] Product name SnapSeek as display name (iOS CFBundleDisplayName, Android android:label, in-app title); bundle id / folder / Melange model name unchanged.
- [ ] Android release gauntlet (LESSONS.md): AGP 8.9.1 / Kotlin 2.1.0 / Gradle 8.11.1, minify+shrink off, legacy jniLibs packaging, INTERNET + ACCESS_NETWORK_STATE + CAMERA in the MAIN manifest, suppressUnsupportedCompileSdk.
- [ ] iOS release build: NSCameraUsageDescription, iOS 16.6 min, flutter build ios --release --no-codesign.
- [ ] **[BLOCKED – human/GATE-0]** Inject registered Melange name/version for BOTH models into model_registry.dart once the dashboard paste-back arrives (expected ajayshah/VisualSearchDetect v1, ajayshah/VisualSearchEmbed v1). Build proceeds against placeholders meanwhile.
- [ ] **[BLOCKED – human]** Physical-device run (iOS/Android release), read served target+apType from native console, confirm no first-inference MPSGraph crash on the FastViT embed tower (iOS 26.3+).

## Deliverables

- Flutter source under apps/VisualSearch/Flutter/ (screens, dual-model MelangeService, preprocessor, detector postprocessor, global NMS, embed preprocessor, gallery ranking, models, widgets/overlay/HUD).
- Late-binding constants: lib/services/model_registry.dart (both models' name/version placeholders — the only file referencing them).
- Bundled demo gallery assets: precomputed 512-d vectors (from the real exported ONNX) + thumbnails.
- Tier A test suite + test/benchmark/hot_path_benchmark.dart with Tier B before/after deltas.
- Custom launcher icon (assets/icon/app_icon.png) + SnapSeek product name.
- Android + iOS release build configuration carrying all LESSONS.md fixes.
- This HANDOFF.md (living ticket), finalized at GATE 3.

## Validation

Tier A test files (hand-built data, known expected output) mapped to the spec Validation focus:

- decode_channel_major_test.dart — [1,84,8400] channel-major decode (anchor-stride not channel-stride), one-anchor + high-index-anchor + wrong-length-buffer assert.
- letterbox_inverse_roundtrip_test.dart — forward-letterbox a known box then invert; returns to original within tolerance across several source sizes + geometry constants.
- crop_space_mapping_test.dart — the crop rect used for embed is in original-frame px (not letterbox space), margin expansion clamped to bounds.
- global_nms_test.dart — class-agnostic global NMS: two overlapping boxes of different classes collapse to one (contrast with per-class).
- threshold_boundary_test.dart — just-below 0.25 dropped, just-above kept.
- embed_preprocessing_test.dart — plain /255 with NO ImageNet mean/std; output unit-norm (‖v‖≈1) and cosine(v,v)=1 via the dot-product path.
- no_detection_fallback_test.dart — zero-box output → center-crop fallback, embed never skipped.
- orientation_roundtrip_test.dart — chosen transform round-trips a known box for the believed buffer orientation.
- gallery_ranking_test.dart — dot-product top-K ranking correct + descending; identical vector scores 1.0.

Tier C (device-only, filled at GATE 3): served artifact expectation, modelMode (RUN_AUTO), device-console command, FastViT MHSA MPSGraph-crash watch on iOS 26.3+, signing/build-config, network/cold-start, run-N-times acceptance.

## References

- App directory: apps/VisualSearch
- Core SDK: ZETIC Melange (zetic_mlange ^1.8.1, Flutter FFI) — dual-model
- Model 1 (detector): Ultralytics YOLO11n COCO — input float32[1,3,640,640] RGB letterboxed /255, output float32[1,84,8400] channel-major, NMS not baked in, 80 COCO classes (used as salient-object localizer)
- Model 2 (embedding): MobileCLIP2-S0 image tower (timm fastvit_mci0.apple_mclip2_dfndr2b) — input float32[1,3,256,256] RGB /255 no mean/std, output float32[1,512] L2-normalized in-graph (license: apple-amlr — demo only)
- Gallery source: Marqo/deepfashion-inshop (60 real product photos, 16 instances × 3-4 views; research-use license — demo only)
- Reference apps studied: FireDetectionYOLO (PyroGuard) + SafetyPPEYOLO (SiteGuard) for YOLO pipeline/tests; no built dual-model app exists in-repo (LiveDocRedact/SignTranslate/VoxScribe are empty stubs) — dual lifecycle designed from the single-model pattern instantiated twice
- Test device: TBD (human, GATE 3)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
