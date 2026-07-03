Goal
A real-time, fully-offline scene-text reader ("GlyphGo") for Flutter (iOS),
powered by a two-model PP-OCRv5 pipeline (DBNet detector + SVTR-LCNet CTC
recognizer) through the ZETIC Melange SDK. Point the phone at a street sign,
menu, or label and the text is read live, pinned to the regions as the camera
moves — the pitch is a traveler with no signal / no roaming data. Detector
finds arbitrary-oriented text quads; Dart deskews and crops each region; the
recognizer greedy-CTC-decodes each crop; strings overlay live with an IoU-keyed
cache so the demo stays smooth on a CPU-fallback artifact. The optional local
translate step is OMITTED from v1 (GATE-2 decision: no model exists and a
credible FR/ES/DE/IT/PT phrase table is not trivial; the offline-reading demo
stands alone).

Todo List
[x] Export text DETECTOR (PaddlePaddle/PP-OCRv5_mobile_det) to static-shape ONNX
    (opset 12, [1,3,736,736] -> [1,1,736,736] prob map, Sigmoid baked); artifact
    ppocrv5_mobile_det.onnx (~4.75 MB) + ppocrv5_mobile_det_sample_input.npy.
[x] Export text RECOGNIZER (PaddlePaddle/latin_PP-OCRv5_mobile_rec, SVTR-LCNet
    CTC) to static-shape ONNX (opset 12, [1,3,48,320] -> [1,40,838], Softmax
    baked); artifact latin_ppocrv5_mobile_rec.onnx (~8.0 MB) +
    latin_ppocrv5_mobile_rec_sample_input.npy.
[x] Ship CTC dictionary latin_charset.txt (836 chars, order preserved) — Dart
    prepends blank@0 and appends space@837 to rebuild the 838-class map.
[x] Author model_selection.md (scene-text detector + scene-text recognizer
    shortlists, winner rationale) and SPEC_stub.md (full pre/post-proc spec).
[x] GATE 0 RESOLVED — both models registered on Melange (2026-07-02):
    - ajayshah/SignTranslate_Detect v1 (NOTE: replaces the proposed
      "SceneTextDetector"). Served shapes CONFIRMED = export: input x
      float32[1,3,736,736] NCHW BGR -> output fetch_name_0 float32[1,1,736,736]
      prob map, Sigmoid baked. Dashboard bench: NPU 3.80/7.35/12.86 ms
      (min/med/avg), GPU med 219 ms, CPU med 169.1 ms; deployability 98%,
      FP32 100%. CAVEAT: accuracy row reads -4.12..0.00 dB (anomalous — likely
      SNR degenerating on a heatmap output, same artifact as LiveDocRedact's
      detector); MUST verify a non-degenerate heatmap on-device (HUD
      min/max/mean) before trusting results.
    - ajayshah/SignTranslate_Rec v1 (NOTE: "_Rec", NOT "_Recognize"; replaces
      the proposed "SceneTextRecognizer"). Served shapes CONFIRMED = export:
      input x float32[1,3,48,320] NCHW BGR -> output fetch_name_0
      float32[1,40,838] time-major, Softmax baked. Dashboard bench: NPU
      0.51/1.31/4.02 ms, GPU med 49.0 ms, CPU med 32.4 ms; accuracy
      13.25–29.18 dB (healthy); deployability 98%, FP32 100%.
    - SDK create(name:) strings are exactly "ajayshah/SignTranslate_Detect" /
      "ajayshah/SignTranslate_Rec" WITH the slash ("ZETIC |" in the dashboard
      header is a display prefix, not the account). Version 1 assumed (first
      upload) — confirm at first SDK create. modelMode RUN_AUTO for both.
      Benchmarked != served: budget for CPU fallback (~169 ms det +
      ~32 ms/crop) until the console shows otherwise.
[ ] Flutter scaffold under apps/SignTranslate/Flutter/: pubspec (zetic_mlange
    pinned 1.8.1, camera, flutter_launcher_icons dev-dep), theme, main.dart,
    latin_charset.txt bundled as a Flutter asset, iOS/Android platform config
    (NSCameraUsageDescription, iOS 16.6 min, minSdk 24).
[ ] Key handling: personal key injected via --dart-define MELANGE_PERSONAL_KEY
    (String.fromEnvironment); NO key material in the repo; dedicated on-screen
    error state (with run instructions) if the define is missing.
[ ] MelangeService for BOTH models: sequential create (detector then
    recognizer) with staged download progress on the loading screen, one dummy
    warm-up inference EACH after load, close() both on dispose. Copy every
    run() output via Float32List.fromList before the next run — asFloat32List()
    is a view over a reused native buffer (critical for the per-crop recognizer
    loop).
[ ] Long-lived pipeline isolate (no per-frame compute() double-spawn): all
    heavy Dart (BGR conversion, letterbox-736 preprocess, DB postprocess, quad
    deskew, rec preprocess, CTC decode, tracker update) runs in ONE persistent
    isolate; frame bytes cross the boundary once per processed frame; both
    model.run calls stay on the main isolate (model handles bind there).
    Pre-allocated input Float32Lists ([1,3,736,736] and [1,3,48,320]).
[ ] Detector preprocessing: source pixel format -> BGR (drop alpha, keep BGR —
    no RGB swap anywhere), letterbox-resize to 736x736 preserving aspect
    (record scale + pad offsets for the exact inverse), /255 then per-channel
    ImageNet normalize mean [0.485,0.456,0.406] std [0.229,0.224,0.225] in BGR
    channel order 0,1,2, NCHW into the pre-allocated Float32List.
[ ] Detector DB post-processing in Dart: binarize prob map at 0.3 (NO extra
    sigmoid — baked), connected components (8-connectivity, iterative flood
    fill), drop regions with mean prob < box_thresh 0.6, min-area rotated box
    (convex hull + rotating calipers), unclip by ratio 1.5 (offset
    d = area*1.5/perimeter), min-size filter, undo letterbox to frame space,
    order quads top->bottom then left->right (row-band grouping).
[ ] Per-crop quad deskew: 4-point homography (quad -> upright rect, 8x8 linear
    solve, no external dep) with bilinear sampling from the retained BGR frame;
    dest size from quad edge lengths; PaddleOCR-parity rotate-90 for tall crops
    (h >= 1.5w).
[ ] Recognizer preprocessing per crop: aspect-preserving resize to height 48,
    width = min(round(48*w/h), 320), normalize (px/255 - 0.5)/0.5 -> [-1,1],
    right-pad the NORMALIZED tensor with 0.0 to width 320 (pad, never
    stretch — PaddleOCR pads post-normalization), NCHW [1,3,48,320].
[ ] Greedy CTC decode in Dart: charset map asserted to be EXACTLY 838 entries
    (blank@0 + latin_charset.txt lines 1..836 + space@837 — NOT 438, no other
    app's dictionary); per-step argmax over the LAST axis (C=838) stepping
    T=40; collapse consecutive duplicates; drop blank; NO extra softmax
    (baked); confidence = mean max-prob over kept non-blank steps.
[ ] RegionTracker (IoU-keyed staggered cache): match new quads to cached
    entries by bbox IoU >= 0.5; hit -> reuse cached string WITHOUT re-running
    the recognizer and update the stored quad (overlay tracks motion); miss ->
    recognition candidate; evict entries unmatched for N detection cycles
    (default 8, tunable const). Cache empty-string results too (outline-only
    display) so false-positive regions don't churn budget.
[ ] FrameScheduler (all five SPEC-mandated behaviors): (1) top-K recognition
    per frame, K=3 default tunable, priority = cache-misses first then larger
    area; (2) the IoU cache above; (3) adaptive detection cadence from an EMA
    of measured detector ms (every frame when fast; interval scales with
    emaDetMs on CPU fallback; cached overlays redrawn between detection
    frames); (4) _busy guard — drop, never queue, frames while a pass is in
    flight; (5) HUD readouts (below). Injectable clock for tests.
[ ] Live UI: text overlay pinning decoded string + confidence to each quad
    (repaint only when results change); HUD with detector ms (last + EMA),
    recognizer ms/crop, crops-run-this-frame, regions-read count, buffer WxH,
    detector heatmap min/max/mean (the accuracy-anomaly check), FPS; prominent
    "works offline / no signal needed" badge.
[ ] Orientation handling: measure the real buffer WxH on-device and HUD it; do
    NOT assume a landscape buffer (PyroGuard's bug was a SPURIOUS rotation on
    an already-upright buffer); rotationDegrees plumbed for Android sensor
    orientation; overlay mapping is a pure, unit-tested frame->screen function
    (BoxFit.cover math); round-trip a known quad before trusting the overlay.
[ ] Branding: display name "GlyphGo" (iOS CFBundleDisplayName, Android
    android:label, MaterialApp title, app-bar/loading text — bundle id, folder,
    Melange names unchanged); custom 1024x1024 travel/glyph launcher icon
    (waypoint/street-sign motif with a stylized glyph) generated to
    assets/icon/app_icon.png and applied via flutter_launcher_icons
    (remove_alpha_ios: true).
[ ] Tier-A unit tests (10 files, per the GATE-2 test list): CTC 838-class
    off-by-one + merge semantics; time-major [1,40,838] decode + no-extra-
    softmax; rec preprocess pad-not-stretch (+pad-is-0.0, width cap, [-1,1],
    BGR); detector preprocess (BGR ImageNet norm, NCHW, buffer reuse);
    letterbox-736 inverse round-trip; DB postprocess (0.3/0.6/1.5 boundaries,
    no-extra-sigmoid discriminator, components, rotated box, reading order,
    coordinate spaces); quad deskew (homography + deskew-equivalence +
    rotate-90 rule); dual orientation (buffer transform round-trip + overlay
    mapping); RegionTracker (hit/miss/boundary/eviction, recognizer-call spy);
    FrameScheduler (K-cap, priority, fake-clock cadence, _busy dropping).
[ ] A4 hot-path micro-benchmark (test/benchmark/hot_path_benchmark.dart): mock
    720x1280 frame -> detector preprocess + DB postprocess on a synthetic
    ~6-region heatmap; 3 crops (realistic K) of deskew + rec preprocess + CTC
    decode on mock [1,40,838] tensors; report medians (per-stage + per-frame)
    over >= 50 iterations as the Tier-B baseline. Excludes model.run.
[ ] Tier-B optimization pass: each lever measured against the A4 baseline
    (>= 0.5% or removed) — fused single-pass preprocess, threshold-before-
    geometry decode, pre-allocated buffers, typed-data views, overlay repaint
    gating; isolate/copy costs already addressed by design (long-lived isolate,
    single frame-byte crossing).
[ ] Tier-A gates: flutter analyze zero errors/warnings, no TODOs/stubs;
    release-mode device build config compiles (iOS release path).
[ ] GATE 3: finalize this ticket + validation report + Tier C checklist
    (served-artifact console readout for BOTH models, detector heatmap
    non-degeneracy check, modelMode honesty, signing/OS gates, release-build
    note, double first-launch download + fresh-install rehearsal,
    non-determinism acceptance, embedded-key note). Human device run.

Deliverables
- Flutter source under apps/SignTranslate/Flutter/ (loading + live camera
  screens, dual-model MelangeService, pipeline isolate, detector preprocessor,
  DB postprocessor, quad deskew, recognizer preprocessor, CTC decoder,
  RegionTracker, FrameScheduler, text overlay + HUD widgets, GlyphGo branding
  + launcher icon) — PENDING (post-GATE-2).
- Model assets PRESENT: export.py, ppocrv5_mobile_det.onnx (+ sample input),
  latin_ppocrv5_mobile_rec.onnx (+ sample input), latin_charset.txt (836
  chars), model_selection.md, SPEC_stub.md, SPEC.md (FINAL), melange_upload.md.
- Melange registrations DONE: ajayshah/SignTranslate_Detect v1 and
  ajayshah/SignTranslate_Rec v1, served shapes confirmed identical to export.
- Diagnostics: this HANDOFF.md (living ticket), on-HUD buffer WxH + per-stage
  latency + heatmap min/max/mean lines (release-build Dart logs do not reach
  the native console).

References
- App directory: apps/SignTranslate
- Core SDK: ZETIC Melange (zetic_mlange 1.8.1, Flutter FFI) — verified surface:
  await ZeticMLangeModel.create(personalKey:, name:, version:, modelMode:,
  onProgress:) / run(List<Tensor>) with Tensor.float32List / asFloat32List()
  (view over reused native buffer — copy before next run) / close(). See
  apps/FireDetectionYOLO/Flutter/lib/services/melange_service.dart.
- Model A — text DETECTOR: PaddlePaddle/PP-OCRv5_mobile_det (DBNet +
  MobileNetV3, Apache-2.0). float32[1,3,736,736] NCHW BGR, ImageNet norm ->
  float32[1,1,736,736] prob map (Sigmoid baked). DBPostProcess in Dart
  (binarize 0.3, box_thresh 0.6, unclip 1.5). REGISTERED:
  ajayshah/SignTranslate_Detect v1, modelMode RUN_AUTO. CPU med 169 ms / NPU
  med 7.35 ms. Accuracy-row anomaly (-4.12..0 dB) — verify heatmap on-device.
- Model B — text RECOGNIZER: PaddlePaddle/latin_PP-OCRv5_mobile_rec
  (SVTR-LCNet CTC, Latin, Apache-2.0). float32[1,3,48,320] NCHW BGR,
  (px/255-0.5)/0.5 -> float32[1,40,838] time-major (Softmax baked). 838 CTC
  classes: 0=blank, 1..836=latin_charset.txt, 837=space. Greedy CTC in Dart.
  REGISTERED: ajayshah/SignTranslate_Rec v1, modelMode RUN_AUTO. CPU med
  32.4 ms/crop / NPU med 1.31 ms.
- Latency plan: CPU fallback assumed (~169 + K*32 ms) -> top-K=3, IoU>=0.5
  staggered cache, adaptive detection cadence, _busy dropping, HUD readouts
  (all five SPEC-mandated).
- Export: paddle2onnx (opset 12) -> onnxslim static shapes; single OCR-family
  recipe (see export.py, model_selection.md).
- Frameworks: Flutter, camera plugin, CoreML / Apple Neural Engine (via
  Melange), PaddleOCR / paddle2onnx (export). Translate step: OMITTED from v1
  (no model; not trivially credible).
- Platform: iOS 16.6+, Android minSdk 24. OS traps: FP32-GPU CoreML MPSGraph
  crash on iOS/macOS 26.3+ (not client-fixable — ZETIC filters server-side);
  release builds on device; simulator dead end; TWO first-launch downloads;
  non-determinism acceptance (multiple cold starts + fresh install).
- Key: personal key via --dart-define=MELANGE_PERSONAL_KEY (never committed).
- Test device: TBD (human-owned; iPhone 15 / iOS 26.5 used for PyroGuard).
