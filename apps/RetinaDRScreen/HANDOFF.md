# HANDOFF — RetinaDRScreen (display name: FundusGate)

> Living Jira ticket / plan-of-record. Finalized at GATE 3 (device handoff).
> `[x]` = done, `[ ]` = open, `[ ] [BLOCKED – owner]` = cannot resolve app-side.
> Status: **GATE 3 — READY FOR DEVICE.** Tier A green, Tier B applied, Tier C
> surfaced below. Not merged to main. "Ready for device," never "done."

## Goal

A fully on-device, offline diabetic-retinopathy SCREENER for Flutter (iOS-first), powered by a MobileNetV2-1.4 binary classifier through the ZETIC Melange SDK. The user picks a color fundus image (gallery) or one-taps a bundled sample; the app runs a single on-device forward pass and shows a binary **REFERABLE / NOT-REFERABLE** verdict with a `P(referable)` confidence bar (fixed 0.5 threshold marked), the screened image, and a per-inference latency readout. The image never leaves the device (zero uploads). This is a CLASSIFIER, not a detector: no boxes, no NMS, no letterbox, no anchors. A required non-diagnostic disclaimer is always visible. Capability/latency proof, NOT a validated diagnostic device and NOT a 0–4 severity grader (that is the sibling app RetinaDRGrade).

## Todo List

### Stage 0 — model/export (DONE, upstream)

- [x] Select model via 6-way bakeoff — winner `EscvNcl/MobileNet-V2-Retinopathy` (MobileNetV2-1.4, native binary NRDR/RDR head; smallest, best healthy-eye specificity, only candidate 6/6 on grade-0).
- [x] Export ONNX `mobilenetv2-dr-referable.onnx` (opset 12, static [1,3,224,224], raw logits, no softmax baked in; checker + torch↔ORT parity verified).
- [x] Register on Melange as **RetinaDRScreen v1**, status READY.
- [x] Validate demo images: g0→NOT-REFERABLE P≈0.0000 [10.11,−0.66]; g3→REFERABLE P≈0.9958 [−2.72,2.75]; g4→REFERABLE P≈0.9900 [−2.30,2.29].
- [x] Fix stale `ajayshah/RetinaDRScreen` → `RetinaDRScreen` in melange_upload.md.

### Build (DONE)

- [x] HANDOFF.md created as first artifact; kept updated through the build.
- [x] Scaffold Flutter project (`Flutter/`, org com.zetic, pkg retinadrscreen). Display name "FundusGate" (iOS CFBundleDisplayName, Android android:label, MaterialApp title, loading/app-bar text). Bundle id `com.zetic.retinadrscreen`, folder, and Melange name unchanged.
- [x] `lib/config/secrets.dart` GITIGNORED (verified via `git check-ignore` BEFORE writing) + committed `secrets.example.dart` placeholder template. Real key lives only in the gitignored file; absent from every tracked file, HANDOFF, and doc.
- [x] MelangeService: `create(personalKey, name:'RetinaDRScreen', v1, RUN_AUTO)` → warm-up dummy inference → `run([Tensor])` → `close()`. Inference runs on the model's creating (UI) isolate; image decode is pushed off-isolate via `compute()`.
- [x] Preprocessor: EXIF bake → decode RGB (alpha dropped) → resize shortest-edge 256 (bilinear) → center-crop 224 → ÷255 → (v−0.5)/0.5 → NCHW [1,3,224,224] Float32List.
- [x] Postprocessor: logits[1,2] → softmax ONCE (numerically stable) → P(referable)= softmax[1] → referable if ≥ 0.5 → ScreeningResult.
- [x] Models: ScreeningResult, SampleFundus (3 bundled samples), InferenceOutcome.
- [x] Widgets: verdict banner, P(referable) confidence bar w/ fixed-0.5 marker, offline badge, diagnostics HUD (latency + raw logits + tensor shape), REQUIRED non-diagnostic disclaimer (always visible on the result surface).
- [x] Screens: loading (download progress + warm-up) + main (pick / 3 one-tap samples / result).
- [x] Bundle the 3 validated demo images as assets; wire image_picker gallery flow.
- [x] Custom fundus/retina launcher icon (teal ring + optic disc + vessels) via flutter_launcher_icons (iOS + Android, remove_alpha_ios). Generated from `tool/generate_icon.dart` → `assets/icon/app_icon.png`.
- [x] iOS 16.6 deployment target (pbxproj + Podfile), Android minSdk 24, NSPhotoLibraryUsageDescription.
- [x] Tier A battery (28 tests) + A4 hot-path micro-benchmark. `flutter analyze` clean.
- [x] Tier B: applied the getBytes preprocessing optimization (measured delta below).
- [x] A2 build: `flutter build ios --no-codesign --release` succeeds (Runner.app 50.1 MB, custom icon baked into Assets.car).

### Blocked / human-owned (GATE 3+)

- [ ] **[BLOCKED – human]** Physical-device run: read served target+apType from the native console, confirm NPU vs CPU, signing / Developer Mode / "Always Allow". See Tier C.
- [ ] **[BLOCKED – ZETIC backend]** Confirm the served artifact is not FP32-GPU CoreML on iOS/macOS 26.x (MPSGraph crash class). Lower-risk here (plain CNN, no attention) but not client-fixable; verify on device.
- [ ] **[BLOCKED – legal]** LICENSE pre-ship gate: `EscvNcl/MobileNet-V2-Retinopathy` is `license: other` (undeclared terms). Clear before GTM. Not app-side.
- [ ] Android device run verification once iOS is confirmed.

## Validation report

### Tier A — autonomous gates (ALL GREEN)

- **A1 analyze:** `flutter analyze` → "No issues found!" (0 errors, 0 warnings).
- **A2 build:** iOS release device build succeeds (`flutter build ios --no-codesign --release`, Runner.app 50.1 MB). Custom launcher icon present (not default Flutter). Display name "FundusGate" set on iOS + Android + in-app.
- **A3 unit tests:** 28 tests, all passing. Coverage:
  - Softmax correctness: P0+P1=1, matches analytic 2-class sigmoid, argmax↔larger logit, NOT double-applied, numerically stable at ±1000 logits, rejects wrong-length input.
  - Threshold boundary at 0.5: equal logits (P=0.5) → REFERABLE (≥), just-below → NOT, just-above → REFERABLE.
  - Label mapping (0=Nrdr, 1=Rdr, not inverted): bigger Nrdr → NOT-REFERABLE low P; bigger Rdr → REFERABLE high P; confidence = max(P0,P1).
  - Anti-degeneracy: grade-0 demo logits [10.11,−0.66] → NOT-REFERABLE, P≈0.
  - Normalization exactness: (v−0.5)/0.5 maps 0→−1, 127.5→0, 255→+1; explicitly NOT plain ÷255, NOT ImageNet mean/std.
  - Resize/crop geometry: shortest-edge→256 preserves aspect (512×256, 1000×500→512×256, 500×1000→256×512, 640²→256²); never squashed to 224×224; center-crop origin correct.
  - Channel order RGB (not BGR): R,G,B constants land in NCHW channels 0,1,2; all values in [−1,1]; tensor length 3·224·224.
  - Demo integration harness: real Dart preprocessing runs end-to-end on all 3 real fundus files → well-formed [1,3,224,224] in [−1,1]; postprocessor reproduces the measured decisions/probs within 5e-3 (g0 NOT-REF P≈0.0000, g3 REF P≈0.9958, g4 REF P≈0.9900). (Real ONNX inference is device-only — GATE 3.)
- **A4 micro-benchmark:** full pure-Dart hot path (decode + resize + crop + normalize + softmax + threshold) on a 640×480 mock PNG, 50 iters → **median ~20.1 ms**. This is the Dart post-processing budget, NOT end-to-end device latency (NPU/CPU inference is fixed by Melange and only appears on hardware). Split: PNG decode ~16.3 ms dominates; the resize→crop→normalize loop ~3.6 ms; softmax+threshold negligible.

### Tier B — optimization log (the 0.5% rule)

- **APPLIED — flat RGB buffer vs per-pixel getPixel():** replaced the `cropped.getPixel` loop with a single `cropped.getBytes(order: rgb)` typed-buffer pass. Measured on the A4 harness: the pixel loop dropped **4.07 ms → 3.62 ms (−0.45 ms, ~11% of the loop, ~2.2% of the total hot-path budget)** — above the 0.5% bar, so kept. Also drops alpha cleanly in the same call.
- **Pre-allocated fused single pass:** input Float32List is allocated once per screening and filled in one pass (resize+crop+normalize+NCHW fused); no intermediate per-channel buffers. (One-shot upload app — no per-frame allocation churn to remove.)
- **Model warm-up:** one dummy inference right after load so the first real screening is not the cold one.
- **Decode off the UI isolate:** `compute(preprocessFundusBytes, …)` keeps the ~16 ms PNG decode off the UI thread; only the small [1,3,224,224] result crosses back, and `run` stays on the model's creating isolate.
- **SKIPPED (justified):** per-frame levers (isolate reuse per frame, `_busy` frame guard, repaint throttling, cheapest camera format) — N/A: this is a one-shot still-image classifier with no camera/frame loop. Decode itself (~16 ms, the dominant cost) is the user's picked file and cannot be optimized away without changing image quality.

### Tier C — runtime-risk checklist (surfaced, not tested — the honest 70%)

- **Served artifact.** Expected: FP32 (100% deployable per dashboard); benchmark headline NPU median ~0.95 ms, realistic non-crashing fallback CPU ~20 ms. The client CANNOT force backend/precision — only modelMode reaches the selector. Read the ACTUAL served `target`+`apType` from the native console (`runtimeApType=CPU/NPU`); that, not the dashboard row, is truth. Plain CNN (no attention) → lower-risk for the FP32-GPU CoreML/MPSGraph crash class than ViT/YOLO, but still confirm it is not GPU on iOS/macOS 26.x. If it crashes in MPSGraph on a new OS, escalate to ZETIC to filter GPU server-side (not client-fixable).
- **modelMode.** RUN_AUTO. Do not expect any mode to steer off a crashing artifact.
- **Native observability.** Watch during the run: `xcrun devicectl device process launch --console --terminate-existing --device <UDID> com.zetic.retinadrscreen`. Dart `print`/`debugPrint` does NOT reach this console on a release build — so per-inference latency, the two raw logits, and the input tensor shape are surfaced on the on-screen **DIAGNOSTICS HUD**, not logged.
- **Signing & OS gates (manual, non-scriptable).** Bundle id `com.zetic.retinadrscreen`; set a signing team; Developer Mode ON; trust the profile ("Always Allow"); iOS 16.6+. Photo-library permission string is set (NSPhotoLibraryUsageDescription).
- **Build config.** Use a **release** device build (debug hangs on launch on recent iOS/Xcode; icon-tap of a debug build shows the "launch from Flutter tooling" screen — expected). Simulator is a dead end (device-only xcframework slice).
- **Network & cold start.** The model downloads on first launch over the network (poor Wi-Fi = spinner); the loading screen shows download progress, then warm-up. Rehearse a fresh install + pre-download before the booth.
- **Non-determinism acceptance.** Server-side selection can return a different artifact minute to minute. "It ran once" is not evidence — accept only after clean runs across multiple cold starts and at least one fresh install; re-verify after any backend re-target.
- **Secrets.** The personal key is embedded in the client (gitignored `secrets.dart`, not tracked). Rotate/scope appropriately for distribution.
- **License.** Pre-ship legal gate (undeclared `license: other`) — clear before GTM.

## Deliverables

- Flutter source under `apps/RetinaDRScreen/Flutter/lib/` (main, theme, screens ×2, services: melange_service/preprocessor/postprocessor, models ×3, widgets ×5, config: secrets.example + gitignored secrets).
- Tests: `test/postprocessor_test.dart`, `test/preprocessor_test.dart`, `test/demo_integration_test.dart`, `test/benchmark/hot_path_benchmark.dart` (28 passing).
- Icon tooling: `tool/generate_icon.dart` → `assets/icon/app_icon.png` → generated iOS AppIcon set + Android mipmaps.
- Bundled assets: 3 validated demo fundus images.
- Model assets (Stage 0): export.py, mobilenetv2-dr-referable.onnx, sample_input.npy, registered Melange model RetinaDRScreen v1.
- This finalized HANDOFF.md.

## References

- App directory: apps/RetinaDRScreen (Flutter project under `Flutter/`)
- Core SDK: ZETIC Melange `zetic_mlange` 1.8.1 (Flutter FFI; vendored ZeticMLange.xcframework via CocoaPods; note: no Swift Package Manager support — Flutter 3.44 auto-falls-back to pods, benign warning).
- Model: MobileNetV2-1.4 binary DR screener — `EscvNcl/MobileNet-V2-Retinopathy` (input float32[1,3,224,224] NCHW RGB; output float32[1,2] raw logits; labels 0=Nrdr/not-referable, 1=Rdr/referable). Preprocessing (v−0.5)/0.5; softmax in Dart.
- Spec: apps/RetinaDRScreen/SPEC.md · model_selection.md · demo_images/DEMO_IMAGES.md
- Doc set: apps/agentic-workflow-docs/ (CLAUDE.md, AGENTS.md, VALIDATION.md)
- Reference impl (structure only): apps/FireDetectionYOLO
- Toolchain: Flutter 3.44.3 / Dart 3.12.2, CocoaPods 1.16.2
- Test device: **UNKNOWN** (to be set by human at the device run).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
