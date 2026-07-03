## Goal

A real-time, fully on-device worker-safety PPE compliance demo for Flutter (iOS-first), powered by a YOLOv8s PPE detector (13 classes, 4 rendered) through the ZETIC Melange SDK. Product name: **SiteGuard**. Streams the live camera feed, runs detection each frame on-device, and overlays worn-vs-violation color-coded boxes (Hardhat / NO-Hardhat / Safety Vest / NO-Safety Vest) with a live latency + per-class count HUD. Built to tolerate a ~400 ms CPU-served artifact (frame-drop guard) with ~5 ms NPU as the upside.

## Todo List

- [x] Stage 0: 6-way validation-gated model selection on a 40-image/303-box GT set (winner: ayushgupta7777/safetyvision-yolov8 v2/best.pt, YOLOv8s) — model_selection.md.
- [x] Stage 0: export safetyppe-8s.onnx (opset 12, static float32[1,3,640,640] -> [1,17,8400], no NMS) + sample_input.npy + export.py (family recipe).
- [x] Stage 0: curate demo_validation/ overlays + reproducible eval harness (validation/).
- [x] GATE 0: model registered on Melange dashboard as ajayshah/SafetyPPEYOLO v1, READY; served shapes echo the export exactly; modelMode RUN_AUTO. Benchmark: NPU med 5.63 ms / GPU med 98 ms / CPU med 434 ms, 100% deployable. Observation (non-blocking): dashboard showed "Uploaded Input Data: —" for this model; conversion succeeded and shapes echo correctly, so proceeding — but noted here in case a later serve/accuracy anomaly appears.
- [x] Finalize SPEC.md with GATE-0 paste-back values (no TBDs).
- [x] GATE 2: build plan + Tier A test list APPROVED by orchestrator. Rulings: (1) Mask/NO-Mask EXCLUDED, no settings toggle — whitelist is exactly {3,7,9,12}; (2) Android = minSdk 24, YUV420 path, icon/label, COMPILE-VERIFIED only; iOS release device build is the demo target and the hard A2 gate; (3) bundle id approved: com.zeticai.siteguard.
- [x] Secrets wiring: gitignored lib/config/secrets.dart (personal key, NEVER committed) + committed secrets.example.dart placeholder; gitignore verified (git check-ignore hits secrets.dart; git ls-files/status never show it; the real 36-char key appears in NO tracked/untracked file but secrets.dart).
- [x] Create core Flutter structure (loading screen, camera screen, theme, HUD) under apps/SafetyPPEYOLO/Flutter/ (15 lib files, 10 test files).
- [x] Melange lifecycle wrapper (create(personalKey:, name: 'ajayshah/SafetyPPEYOLO', version: 1, modelMode: RUN_AUTO) -> warm-up dummy inference -> Tensor.float32List run -> close). zetic_mlange 1.8.1 API surface: only modelMode reaches the remote selector; target/apType forwarded as hints only.
- [x] Preprocessing: BGRA (iOS) / YUV420 (Android, BT.601) decode, fused single-pass letterbox-640 (pad 0.5) + bilinear resample + /255 + NCHW into a pre-allocated reused Float32List (lib/services/preprocessor.dart).
- [x] Post-processing: channel-major [1,17,8400] decode (pre-gate on the 4 rendered classes, then full-13 argmax, threshold-first before geometry), per-class thresholds (Hardhat .25; Vest/NO-Hardhat/NO-Vest .15), class whitelist {3,7,9,12} — non-whitelisted argmax winner (incl. Person=11) DROPS the anchor, never relabels — un-letterbox inverse, per-class NMS IoU .45.
- [x] Detection overlay (buffer-orientation-measured mapping + cover-fit math) + HUD (latency per stage, per-class counts, buffer WxH debug line toggle).
- [x] Frame flow: _busy frame-drop guard (no queue), long-lived processing path, inline hot path — NO per-frame isolate spawn (PyroGuard ~20 ms/frame lesson).
- [x] Tier A tests: 45/45 GREEN. channel-major decode (hand-built anchor + stride check), letterbox inverse round-trip, per-class vs global NMS, no-double-sigmoid, per-class threshold boundaries, whitelist enforcement (Person never emitted), orientation round-trip, coordinate-space; hot_path_benchmark.dart.
- [x] Tier B optimization pass with measured before/after deltas (see Tier B Log).
- [x] Custom launcher icon (1024x1024 RGB source assets/icon/app_icon.png, flutter_launcher_icons 0.14.3, remove_alpha_ios: true; 1024 appicon slice regenerated) — not the Flutter default.
- [x] Product name "SiteGuard" as display name only: CFBundleDisplayName=SiteGuard, android:label=SiteGuard, MaterialApp title='SiteGuard'; bundle id com.zeticai.siteguard / folder / Melange name unchanged.
- [x] flutter analyze: 0 errors / 0 warnings / 0 infos ("No issues found"). iOS release build (--no-codesign) OK -> Runner.app 36.8 MB, IPHONEOS_DEPLOYMENT_TARGET bumped 13.0->16.6 to match SPEC + Podfile + PyroGuard. Android release assembleRelease OK -> app-release.apk 197.7 MB (AGP pinned 9.0.1->8.9.1, Kotlin 2.3.20->2.1.0, Gradle 9.1.0->8.11.1: zetic_mlange 1.8.1 legacy DSL breaks under AGP 9 — same fix PyroGuard documented).
- [x] GATE 3: this ticket + Tier A results + Tier B log + Tier C runtime-risk checklist finalized; handed to human for physical-device run.
- [ ] **[BLOCKED – human, GATE 3]** Physical iPhone device run (signing, Developer Mode, release build, native-console watch) — human-only by design.
- [ ] **[BLOCKED – ZETIC backend, contingent]** NPU serving: benchmark shows 5.63 ms NPU median but PyroGuard precedent is a TFLITE_FP16/CPU (~400 ms) serve; if CPU-served on device, the NPU ask goes to ZETIC — not client-fixable.
- [ ] Android run verification once iOS is stable (best-effort, PyroGuard precedent) — compile/assemble verified; on-device run deferred with iOS.

## Deliverables

- Flutter source under apps/SafetyPPEYOLO/Flutter/ (screens, MelangeService, preprocessor, postprocessor, NMS, detection model, overlay/HUD, secrets scaffolding with gitignored key).
- Model assets: export.py, safetyppe-8s.onnx + sample_input.npy (on disk, gitignored per repo convention; regenerable via export.py), registered Melange model ajayshah/SafetyPPEYOLO v1 (READY).
- Decision + validation record: model_selection.md, validation/ (harness, GT, results), demo_validation/ overlays, SPEC.md (finalized), this HANDOFF.md.
- Tier A test suite + hot-path micro-benchmark with recorded baseline.

## Tier A results (2026-07-03, Flutter 3.44.3)

- flutter analyze: No issues found (0 errors / 0 warnings / 0 infos).
- flutter test: 45/45 passed. Coverage: channel-major [1,17,8400] decode with a hand-built one-box anchor + anchor-vs-channel stride assertion + wrong-length assert; letterbox inverse round-trip (1280x720 / 720x1280 / 1920x1080, pad 0.5, //2 offsets); coordinate-space (640-px in, normalized-clamped out, degenerate drop); per-class (anti-global) NMS — overlapping Hardhat+Vest both survive, same-class stronger-only; score-semantics — no double sigmoid; per-class threshold boundaries (0.25 helmet / 0.15 vest+violations, just-below/just-above, per-class-not-global); class-whitelist {3,7,9,12} constant + Person=11 outscore drops the anchor; orientation round-trip (iOS rot-0 no spurious rotation, Android sensor-90, cover-fit mapping).

## Tier B optimization log (measured, A4 micro-benchmark: median of 40, mock 1280x720 BGRA frame + [1,17,8400] output, JIT/desktop — relative deltas carry to device, absolute ms do not)

- Fused single-pass preprocessor + pre-allocated reused input tensor (vs naive 2-pass + fresh 4.9 MB Float32List per frame): 5.72 ms shipped vs 6.00 ms naive (~5% faster). The bilinear sampling dominates the loop, so the fused/pre-alloc win here is mostly avoided GC churn from a 4.9 MB/frame allocation — the larger, unmeasured-in-JIT payoff is on-device steady-state (no per-frame major GC). KEPT.
- Threshold-before-geometry decode + pre-gate on the 4 rendered classes before the full-13 argmax (vs naive: full argmax AND box geometry for every one of 8400 anchors, threshold last): 0.06 ms shipped vs 0.33 ms naive (~5.5x / ~82% faster). KEPT (dominant win).
- Per-class NMS via single-pass classId bucketing + pre-computed box areas (areas hoisted out of the O(n^2) inner loop): folded into the decode figure above; correctness (Hardhat+Vest co-survival) is pinned by Tier A, not just speed. KEPT.
- Pre-allocated input buffer (reused across frames): covered by the preprocessor delta above; eliminates a 4.9 MB allocation per frame. KEPT.
- No per-frame isolate/compute() spawn (inline hot path): NOT a micro-benchmark line — it removes PyroGuard's measured ~20 ms/frame spawn+copy tax, an architectural choice validated by precedent rather than re-benchmarked. KEPT.
- _busy frame-drop guard (no frame queue): correctness/latency-stability guard, no steady-state cost to measure. KEPT.
- Full shipped hot path (preprocess + decode + per-class NMS): 5.93 ms A4 BASELINE — the entire pure-Dart budget is dwarfed by the model run (NPU ~5 ms best case, CPU-served ~434 ms fallback), so no further micro-opt clears the 0.5% rule; stop here.

## Tier C runtime-risk checklist (for the human GATE-3 device run)

- Served artifact is ground truth, not the requested mode: read target+apType from the native console. Watch command (bundle id com.zeticai.siteguard): `xcrun devicectl device console --device <UDID> | grep -i "zetic\|mlange\|siteguard"` (or Console.app filtered to the device + process "Runner"). Dashboard bench is NPU ~5.63 ms median but PyroGuard precedent served TFLITE_FP16/CPU (~434 ms); the _busy guard + HUD latency readout tolerate either — "benchmarked ≠ served".
- Cold start: first launch DOWNLOADS the model over the network before the warm-up inference; loading screen must cover it. Subsequent launches use the cache.
- Backend-selection non-determinism: the remote selector may serve different artifacts across runs/devices; the HUD latency line is the on-device truth. iOS 26.3+ FP32-GPU MPSGraph crash is server-side-filtered by ZETIC (RUN_AUTO).
- Orientation: verify real buffer WxH on the HUD debug line before trusting the overlay (PyroGuard: iOS BGRA arrived upright 720x1280; the bug was a SPURIOUS 90° rotation). iOS passes rotation 0; Android passes sensor orientation.
- Release-only: simulator is a dead end (device-only xcframework slice); Dart prints don't reach the release console — HUD diagnostics only.

## Non-blocking observations (recorded, not blocking GATE 3)

- GATE-2 ruling: Mask/NO-Mask EXCLUDED with no settings toggle; whitelist is exactly {3 Hardhat, 7 NO-Hardhat, 9 NO-Safety Vest, 12 Safety Vest}.
- Dashboard showed "Uploaded Input Data: —" for this model at registration; conversion succeeded and served shapes echo the export exactly, so proceeding — flagged here only in case a later serve/accuracy anomaly appears.

## References

- App directory: apps/SafetyPPEYOLO
- Core SDK: ZETIC Melange (zetic_mlange, Flutter FFI) — verify installed version
- Model: YOLOv8s PPE — ayushgupta7777/safetyvision-yolov8 v2/best.pt (input float32[1,3,640,640], output float32[1,17,8400] channel-major, no NMS; render classes: Hardhat(3), NO-Hardhat(7), NO-Safety Vest(9), Safety Vest(12); Person(11) degenerate — never rendered)
- Architecture reference: apps/FireDetectionYOLO (PyroGuard) — same family, same pipeline shape, source of the orientation/latency/observability lessons
- License posture: AGPL-3.0 weights + Ultralytics AGPL lineage — flagged in model_selection.md; internal demo use, human decision on any distribution
- Test device: physical iPhone (PyroGuard runs used iPhone 15 / iPhone15,4, iOS 26.5); iOS release builds only; simulator is a dead end

🤖 Generated with [Claude Code](https://claude.com/claude-code)
