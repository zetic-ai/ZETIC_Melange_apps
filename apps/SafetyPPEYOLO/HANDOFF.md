Goal
A real-time, fully on-device worker-safety PPE compliance demo for Flutter
(iOS-first), powered by a YOLOv8s PPE detector (13 classes, 4 rendered) through
the ZETIC Melange SDK. Product name: **SiteGuard**. Streams the live camera
feed, runs detection each frame on-device, and overlays worn-vs-violation
color-coded boxes (Hardhat / NO-Hardhat / Safety Vest / NO-Safety Vest) with a
live latency + per-class count HUD. Built to tolerate a ~400 ms CPU-served
artifact (frame-drop guard) with ~5 ms NPU as the upside.

Todo List
[x] Stage 0: 6-way validation-gated model selection on a 40-image/303-box GT set
    (winner: ayushgupta7777/safetyvision-yolov8 v2/best.pt, YOLOv8s) —
    model_selection.md.
[x] Stage 0: export safetyppe-8s.onnx (opset 12, static float32[1,3,640,640] ->
    [1,17,8400], no NMS) + sample_input.npy + export.py (family recipe).
[x] Stage 0: curate demo_validation/ overlays + reproducible eval harness
    (validation/).
[x] GATE 0: model registered on Melange dashboard as ajayshah/SafetyPPEYOLO v1,
    READY; served shapes echo the export exactly; modelMode RUN_AUTO.
    Benchmark: NPU med 5.63 ms / GPU med 98 ms / CPU med 434 ms, 100% deployable.
    Observation (non-blocking): dashboard showed "Uploaded Input Data: —" for
    this model; conversion succeeded and shapes echo correctly, so proceeding —
    but noted here in case a later serve/accuracy anomaly appears.
[x] Finalize SPEC.md with GATE-0 paste-back values (no TBDs).
[ ] GATE 2: build plan + Tier A test list approved by orchestrator (package
    submitted, awaiting approval).
[ ] Secrets wiring: gitignored lib/config/secrets.dart (personal key, NEVER
    committed) + committed secrets.example.dart placeholder; verify gitignore
    catches it BEFORE first commit of Flutter/.
[ ] Create core Flutter structure (loading screen, camera screen, theme, HUD)
    under apps/SafetyPPEYOLO/Flutter/.
[ ] Melange lifecycle wrapper (create(personalKey:, name: 'ajayshah/SafetyPPEYOLO')
    -> warm-up dummy inference -> Tensor.float32List run -> close); verify
    installed zetic_mlange version's exact API surface before coding against it.
[ ] Preprocessing: BGRA (iOS) / YUV420 (Android) decode, fused single-pass
    letterbox-640 (pad 0.5) + /255 + NCHW into a pre-allocated Float32List.
[ ] Post-processing: channel-major [1,17,8400] decode (threshold-first),
    per-class thresholds (Hardhat .25; Vest/NO-Hardhat/NO-Vest .15), class
    whitelist {3,7,9,12} (Person=11 NEVER rendered — degenerate class),
    un-letterbox inverse, per-class NMS IoU .45.
[ ] Detection overlay (buffer-orientation-measured mapping + BoxFit.cover math)
    + HUD (latency per stage, per-class counts, buffer WxH debug line toggle).
[ ] Frame flow: _busy frame-drop guard (no queue), long-lived processing path,
    no per-frame isolate spawn (PyroGuard lesson: compute() cost ~20 ms/frame).
[ ] Tier A tests (exact list in GATE-2 package): channel-major decode,
    letterbox inverse round-trip, per-class vs global NMS, no-double-sigmoid,
    per-class threshold boundaries, whitelist enforcement, orientation
    round-trip, coordinate-space; test/benchmark/hot_path_benchmark.dart.
[ ] Tier B optimization pass with measured before/after deltas (0.5% rule).
[ ] Custom launcher icon (hardhat+vest glyph, 1024x1024 source,
    flutter_launcher_icons, remove_alpha_ios: true) — not the Flutter default.
[ ] Product name "SiteGuard" as display name only (CFBundleDisplayName,
    android:label, in-app title); bundle id / folder / Melange name unchanged.
[ ] flutter analyze zero errors/warnings; release device build compiles.
[ ] GATE 3: finalize this ticket + validation report + Tier C runtime-risk
    checklist; hand to human for physical-device run.
[ ] [BLOCKED – human, GATE 3] Physical iPhone device run (signing, Developer
    Mode, release build, native-console watch) — human-only by design.
[ ] [BLOCKED – ZETIC backend, contingent] NPU serving: benchmark shows 5.63 ms
    NPU median but PyroGuard precedent is a TFLITE_FP16/CPU (~400 ms) serve;
    if CPU-served on device, the NPU ask goes to ZETIC — not client-fixable.
[ ] Android run verification once iOS is stable (best-effort, PyroGuard
    precedent).

Deliverables
- Flutter source under apps/SafetyPPEYOLO/Flutter/ (screens, MelangeService,
  preprocessor, postprocessor, NMS, detection model, overlay/HUD, secrets
  scaffolding with gitignored key).
- Model assets: export.py, safetyppe-8s.onnx + sample_input.npy (on disk,
  gitignored per repo convention; regenerable via export.py), registered
  Melange model ajayshah/SafetyPPEYOLO v1 (READY).
- Decision + validation record: model_selection.md, validation/ (harness, GT,
  results), demo_validation/ overlays, SPEC.md (finalized), this HANDOFF.md.
- Tier A test suite + hot-path micro-benchmark with recorded baseline.

References
- App directory: apps/SafetyPPEYOLO
- Core SDK: ZETIC Melange (zetic_mlange, Flutter FFI) — verify installed version
- Model: YOLOv8s PPE — ayushgupta7777/safetyvision-yolov8 v2/best.pt
  (input float32[1,3,640,640], output float32[1,17,8400] channel-major, no NMS;
  render classes: Hardhat(3), NO-Hardhat(7), NO-Safety Vest(9), Safety Vest(12);
  Person(11) degenerate — never rendered)
- Architecture reference: apps/FireDetectionYOLO (PyroGuard) — same family,
  same pipeline shape, source of the orientation/latency/observability lessons
- License posture: AGPL-3.0 weights + Ultralytics AGPL lineage — flagged in
  model_selection.md; internal demo use, human decision on any distribution
- Test device: physical iPhone (PyroGuard runs used iPhone 15 / iPhone15,4,
  iOS 26.5); iOS release builds only; simulator is a dead end
