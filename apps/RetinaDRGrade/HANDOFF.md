# HANDOFF — RetinaDRGrade (display name: GradeVue)

> Living Jira ticket / plan-of-record. Created as the first build artifact (after
> GATE 1), finalized at GATE 3 (device handoff).
> `[x]` = done, `[ ]` = open, `[ ] [BLOCKED – owner]` = cannot resolve app-side.
> Status: **GATE 3 — READY FOR DEVICE.** Tier A green, Tier B applied, Tier C
> surfaced below. Not merged to main. "Ready for device," never "done."

## Goal

A fully on-device, offline diabetic-retinopathy SEVERITY GRADER for Flutter (iOS-first), powered by a ViT-base 5-class classifier through the ZETIC Melange SDK. The user picks a color fundus image (gallery) or one-taps a bundled sample; the app runs a single on-device forward pass and shows the full **0–4 DR severity grade** (0 No DR · 1 Mild · 2 Moderate · 3 Severe · 4 Proliferative) with a 5-way per-grade confidence bar, a REFERABLE (grade ≥ 2) flag, the graded image, and a per-inference latency readout. The image never leaves the device (zero uploads). This is a CLASSIFIER, not a detector: no boxes, no NMS, no letterbox, no anchors. A required non-diagnostic disclaimer is always visible. Capability / latency proof, NOT a validated diagnostic device. This app surfaces the full 0–4 grade; the sibling `RetinaDRScreen` ("FundusGate") is the tiny binary referable screener and must not show a severity grade.

## Todo List

### Stage 0 — model/export (DONE, upstream)

- [x] Select model via 6-way bakeoff — winner `Kontawat/vit-diabetic-retinopathy-classification` (ViT-base, 5-class grade head; best exact-grade accuracy 0.667, only model with referable sens 1.00 AND spec 0.833, spans all 5 grades non-degenerately; apache-2.0 clean license).
- [x] Export ONNX `vit-base-dr-grade.onnx` (opset 12, static [1,3,224,224], raw logits[1,5], no softmax baked in; eager attention; checker + torch↔ORT parity).
- [x] Register on Melange as **ajayshah/RetinaDRGrade v1** (account/project slash form — confirmed on-device this run; a bare name throws MlangeException(3)).
- [x] Validate demo images (exported ONNX): g0→0 No DR p≈0.982; g3→3 Severe p≈0.810; g4→4 Proliferative p≈0.809. 3/3 exact; referable sens/spec 1.00 on subset.
- [x] Reconcile Melange name to the slash form `ajayshah/RetinaDRGrade` in SPEC.md / melange_upload.md (the bare-name variant throws MlangeException(3) on-device).

### Build (DONE)

- [x] HANDOFF.md created as first artifact; kept updated through the build.
- [x] Scaffold Flutter project (`Flutter/`, org com.zetic, pkg retinadrgrade). Display name "GradeVue" (iOS CFBundleDisplayName, Android android:label, MaterialApp title, loading/app-bar text). Bundle id `com.zetic.retinadrgrade`, folder, and Melange name unchanged.
- [x] `lib/config/secrets.dart` GITIGNORED (verified via `git check-ignore` BEFORE writing) + committed `secrets.example.dart` placeholder. Real key lives only in the gitignored file; absent from every tracked file, HANDOFF, and doc.
- [x] MelangeService: `create(personalKey, name:'ajayshah/RetinaDRGrade', v1, RUN_AUTO)` → warm-up dummy inference → `run([Tensor])` → `close()`. Inference runs on the model's creating (UI) isolate; image decode pushed off-isolate via `compute()`.
- [x] Preprocessor: EXIF bake → decode RGB (alpha dropped) → **PLAIN resize 224×224 bilinear** (NOT the sibling's shortest-edge-256 → center-crop) → ÷255 → (v−0.5)/0.5 → NCHW [1,3,224,224] Float32List.
- [x] Postprocessor: logits[1,5] → softmax ONCE (numerically stable) → argmax = grade (IDENTITY id2label, no remap) → referable = grade ≥ 2 → GradingResult.
- [x] Models: GradingResult (grade + 5 probs + referable + logits), SampleFundus (3 bundled samples), InferenceOutcome.
- [x] Widgets: grade banner (large "Grade N — Label"), 5-bar per-grade confidence (argmax highlighted, referable threshold marked at grade 2), REFERABLE flag, offline badge, diagnostics HUD (latency + 5 raw logits + softmax vector + tensor shape), REQUIRED non-diagnostic disclaimer (always visible on the result surface).
- [x] Screens: loading (download progress + warm-up, ~328 MB first-launch note) + main (pick / 3 one-tap samples / result).
- [x] Bundle the 3 validated demo images (g0/g3/g4) as assets; wire image_picker flow.
- [x] Custom grading-flavored fundus launcher icon (severity-gradient ring + optic disc + vessels) via flutter_launcher_icons (iOS + Android, remove_alpha_ios). Generated from `tool/generate_icon.dart` → `assets/icon/app_icon.png`.
- [x] iOS 16.6 deployment target (pbxproj + Podfile), Android minSdk 24, NSPhotoLibraryUsageDescription.
- [x] Tier A battery + A4 hot-path micro-benchmark. `flutter analyze` clean.
- [x] Tier B: applied the getBytes fused-single-pass preprocessing optimization (measured delta below).
- [x] A2 build: `flutter build ios --no-codesign --release` succeeds; custom icon baked into Assets.car; display name "GradeVue".

### Blocked / human-owned (GATE 3+)

- [ ] **[BLOCKED – human]** Physical-device run: read served target+apType from the native console, confirm NPU vs CPU/GPU, signing / Developer Mode / "Always Allow". See Tier C.
- [ ] **[BLOCKED – ZETIC backend]** **HIGHEST-risk artifact in the repo:** confirm the served artifact is NOT FP32-GPU CoreML on iOS/macOS 26.x. This is a ViT self-attention graph — exactly the MPSGraph GPU-compiler crash class (GPU median 838 ms / MAX 6.78 s vs NPU ~10 ms). Not client-fixable; if it aborts in MPSGraph, escalate to ZETIC to filter the GPU candidate server-side for that OS.
- [ ] **[BLOCKED – human]** First-launch ~328 MB fp32 download over booth Wi-Fi: rehearse a fresh-install cold start + pre-download/pre-warm before the demo (far heavier than the sibling's ~17 MB).
- [ ] Android device run verification once iOS is confirmed.

## Validation report

### Tier A — autonomous gates (ALL GREEN)

- **A1 analyze:** `flutter analyze` → "No issues found!" (0 errors, 0 warnings).
- **A2 build:** iOS release device build succeeds (`flutter build ios --no-codesign --release`, Runner.app 29.9 MB). Custom launcher icon present (severity-gradient fundus glyph, not default Flutter) — 21 iOS AppIcon PNGs + Assets.car. Display name "GradeVue" baked into the built app (verified CFBundleDisplayName), Android label + MaterialApp title. (SPM warning for zetic_mlange is benign — Flutter auto-falls back to CocoaPods.)
- **A3 unit tests:** 33 tests, all passing across 4 files. Coverage:
  - **Softmax ONCE over 5 logits:** Σ probs = 1, matches explicit exp/normalize, NOT double-applied, numerically stable at ±1000 logits, rejects wrong-length vectors.
  - **argmax → grade IDENTITY id2label:** the largest-logit index IS the grade for every grade 0..4 (no remap); permuting logits moves the grade correspondingly; ties → lowest.
  - **Referable = grade ≥ 2 boundary:** grade 0/1 → NOT referable; grade 2 (exact boundary)/3/4 → referable. Derived from the argmax grade, not a separate probability threshold.
  - **(v−0.5)/0.5 normalization exactness:** maps 0→−1, 127.5→0, 255→+1; explicitly NOT plain ÷255, NOT ImageNet mean/std.
  - **PLAIN resize-224 geometry (key difference from the sibling):** non-square images resize directly to 224×224; a left/right-edge stripe SURVIVES (a center-crop pipeline would discard the edges) — proving no shortest-edge-256 → center-crop.
  - **Channel order RGB (not BGR):** R,G,B constants land in NCHW channels 0,1,2; all values in [−1,1]; tensor length 3·224·224.
  - **Anti-degeneracy / spread + demo integration:** postprocessor reproduces the DEMO_IMAGES.md 5-way softmax within 5e-3 and the exact grade (g0→0 top≈0.982 not-ref; g3→3 top≈0.810 ref; g4→4 top≈0.809 ref); mass stays spread on neighbouring grades (no collapse). Real Dart preprocessing runs end-to-end on the 3 real fundus fixtures (loaded from the repo `demo_images/`, NOT bundled — app is upload-only) → well-formed [1,3,224,224] in [−1,1]. (Real ONNX inference is device-only — GATE 3; synthetic logits = ln(softmax) reproduce the published distributions exactly.)
- **A4 micro-benchmark:** full pure-Dart hot path (decode + plain-resize-224 + normalize + NCHW + softmax + argmax) on a 640×480 mock PNG, 50 iters → **median 17.75 ms** (min 17.35, max 20.35). This is the Dart post-processing budget, NOT end-to-end device latency (ViT NPU/CPU/GPU inference is fixed by Melange and only appears on hardware — EXPECTED NPU ~10 ms / CPU ~598 ms; GPU is a crash path). PNG decode dominates the budget.

### Tier B — optimization log (the 0.5% rule)

- **APPLIED — fused getBytes() single pass vs per-pixel getPixel():** the pixel-read loop uses one `resized.getBytes(order: rgb)` typed-buffer pass with resize+normalize+NCHW fused, instead of a per-pixel `getPixel()` (which allocates a Pixel accessor per call). Measured on a 224×224 image, 200 iters: **getPixel 0.525 ms → getBytes 0.140 ms (−0.385 ms, ~2.2% of the 17.75 ms hot-path budget)** — above the 0.5% bar, so kept. Also drops alpha cleanly in the same call.
- **Pre-allocated fused single pass:** the input Float32List is allocated once per grading and filled in one pass; no intermediate per-channel buffers. (One-shot upload app — no per-frame allocation churn to remove.)
- **Model warm-up:** one dummy inference right after load so the first real grading is not the cold one.
- **Decode off the UI isolate:** `compute(preprocessFundusBytes, …)` keeps the PNG decode off the UI thread; only the small [1,3,224,224] result crosses back, and `run` stays on the model's creating isolate.
- **SKIPPED (justified):** all per-frame levers (isolate reuse per frame, `_busy` frame guard, repaint throttling, cheapest camera format, decode-anchors-once, NMS bucketing) — N/A: this is a one-shot still-image classifier with no camera/frame loop and no boxes / anchors / NMS. Decode itself (the dominant hot-path cost) is the user's uploaded file and cannot be optimized without changing image quality.

### Tier C — runtime-risk checklist (surfaced, not tested — the honest 70%)

- **Served artifact — HIGHEST GPU-crash risk in the repo.** This is a **ViT-base self-attention graph**, exactly the fusion-pattern class that triggers Apple's GPU compiler (MPSGraph) bug: a served FP32-GPU CoreML artifact can load cleanly then abort at the FIRST inference (`MLIR pass manager failed`, SIGABRT, uncatchable in Dart). The dashboard shows the GPU path is catastrophic here (**GPU median 838 ms, MAX 6.78 s**) vs **NPU ~10 ms** / CPU ~598 ms. The client CANNOT force backend/precision — only modelMode reaches the selector. **Read the ACTUAL served `target`+`apType` from the native console (`runtimeApType=…`) and confirm it is NOT FP32-GPU CoreML on iOS/macOS 26.x.** If it crashes in MPSGraph, escalate to ZETIC to filter the GPU candidate server-side for that OS (not client-fixable). Budget the NPU path as the only good one; CPU (~598 ms) is the realistic non-crashing fallback until `runtimeApType=NPU` is confirmed.
- **modelMode.** RUN_AUTO. Do NOT expect any mode to steer off a crashing artifact (all four modes returned the same crashing GPU artifact on PyroGuard).
- **First-launch ~328 MB download (network / cold start).** The served ViT-base model is ~328 MB fp32 (82–328 MB across the 3 quantizations) — far heavier than the sibling's ~17 MB. Over booth Wi-Fi the first-run pull-and-cache is a long, user-visible spinner. The loading screen surfaces download progress + a "~328 MB" note, then warm-up. **Rehearse a fresh-install cold start + pre-download/pre-warm on the real booth network** before the demo so the first run isn't a stall.
- **Native observability.** Watch during the run: `xcrun devicectl device process launch --console --terminate-existing --device <UDID> com.zetic.retinadrgrade`. Dart `print`/`debugPrint` does NOT reach this console on a release build — so per-inference latency, the 5 raw logits, the softmax vector, the predicted grade, and the input tensor shape are surfaced on the on-screen **DIAGNOSTICS HUD**, not logged.
- **Signing & OS gates (manual, non-scriptable).** Bundle id `com.zetic.retinadrgrade`; set a signing team; Developer Mode ON; trust the profile ("Always Allow"); iOS 16.6+. Photo-library permission string is set (NSPhotoLibraryUsageDescription).
- **Build config.** Use a **release** device build (debug hangs on launch on recent iOS/Xcode; a debug icon-tap shows the "launch from Flutter tooling" screen — expected). Simulator is a dead end (device-only xcframework slice; no camera needed anyway — upload-only).
- **Non-determinism acceptance.** Server-side selection can return a different artifact minute to minute. "It ran once" is not evidence — accept only after clean runs across multiple cold starts and at least one fresh install; re-verify after any backend re-target.
- **Secrets.** The personal key is embedded in the client (gitignored `secrets.dart`, verified via `git check-ignore` before writing; committed `secrets.example.dart` placeholder). Rotate/scope appropriately for distribution.
- **License.** `apache-2.0` — clean, NO pre-ship legal gate (a plus over the sibling RetinaDRScreen, whose weights are `license: other` / undeclared).

## Deliverables

- Flutter source under `apps/RetinaDRGrade/Flutter/lib/` (main, theme, screens ×2, services: melange_service/preprocessor/postprocessor, models ×3, widgets ×6, config: secrets.example + gitignored secrets).
- Tests: `test/postprocessor_test.dart`, `test/preprocessor_test.dart`, `test/demo_integration_test.dart`, `test/benchmark/hot_path_benchmark.dart`.
- Icon tooling: `tool/generate_icon.dart` → `assets/icon/app_icon.png` → generated iOS AppIcon set + Android mipmaps.
- Bundled assets: 3 validated demo fundus images (g0, g3, g4).
- Model assets (Stage 0): export.py, vit-base-dr-grade.onnx (not in worktree — Stage-0 artifact), sample_input.npy, registered Melange model ajayshah/RetinaDRGrade v1.
- This finalized HANDOFF.md.

## References

- App directory: apps/RetinaDRGrade (Flutter project under `Flutter/`)
- Core SDK: ZETIC Melange `zetic_mlange` 1.8.1 (Flutter FFI; vendored ZeticMLange.xcframework via CocoaPods).
- Model: ViT-base 5-class DR grader — `Kontawat/vit-diabetic-retinopathy-classification` (input float32[1,3,224,224] NCHW RGB, plain resize-224 + (v−0.5)/0.5; output float32[1,5] raw logits; id2label identity {0..4}; referable = grade ≥ 2). apache-2.0.
- Spec: apps/RetinaDRGrade/SPEC.md · model_selection.md · DEMO_IMAGES.md
- Doc set: apps/agentic-workflow-docs/ (CLAUDE.md, AGENTS.md, VALIDATION.md)
- Reference impl (structure only): apps/RetinaDRScreen (sibling binary screener)
- Toolchain: Flutter 3.44.3 / Dart 3.12.2
- Test device: **UNKNOWN** (to be set by human at the device run).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
