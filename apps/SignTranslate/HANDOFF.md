Goal
A real-time, fully-offline scene-text reader for Flutter (iOS), powered by a
two-model PP-OCRv5 pipeline (DBNet detector + SVTR-LCNet CTC recognizer) through
the ZETIC Melange SDK. Point the phone at a street sign, menu, or label and read
the text live as the camera moves, with an OPTIONAL on-device translate step for
a traveler who has no signal / no roaming data. Detector finds arbitrary-oriented
text quads each frame; Dart de-skews and crops each region; the recognizer
greedy-CTC-decodes each crop to a string overlaid live on the scene.

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
[ ] [BLOCKED – human/dashboard, GATE 0] Register BOTH models on Melange:
    ajayshah/SceneTextDetector (v1) from ppocrv5_mobile_det.onnx +
    ppocrv5_mobile_det_sample_input.npy, and ajayshah/SceneTextRecognizer (v1)
    from latin_ppocrv5_mobile_rec.onnx + latin_ppocrv5_mobile_rec_sample_input.npy.
    Drag each ONNX + sample input into the dashboard, wait for READY, and paste
    back the registered name+version and served input/output shapes for both.
    App is blocked until then (see melange_upload.md).
[ ] Create core Flutter structure: loading screen (dual-model download + warm-up,
    progress bar) and LIVE camera screen with HUD (regions-read count, detector
    ms + recognizer ms, buffer WxH debug line).
[ ] MelangeService lifecycle for BOTH models (create -> Tensor.float32List -> run
    -> close) plus a one-shot warm-up run of each model on first load.
[ ] Detector preprocessing: source pixel format -> BGR (drop alpha, keep BGR),
    letterbox-resize to 736x736 preserving aspect (remember scale + pad offsets),
    ImageNet normalize (mean [0.485,0.456,0.406], std [0.229,0.224,0.225] after
    /255), NCHW [1,3,736,736] Float32List.
[ ] Detector DB post-processing in Dart: binarize prob map at 0.3, contour /
    connected components, box_thresh 0.6 mean-prob filter, min-area rotated box,
    unclip (dilate) by ratio 1.5, undo letterbox to screen space.
[ ] Text-region grouping: emit quads ordered top->bottom, left->right for reading
    order; drive the per-crop recognizer loop.
[ ] Recognizer preprocessing per crop: warp/de-skew angled quad to upright rect,
    aspect-preserving resize to height 48 width=min(round(48*w/h),320), right-pad
    with zeros to width 320, BGR, normalize (pixel/255 - 0.5)/0.5 -> [-1,1], NCHW
    [1,3,48,320] Float32List.
[ ] Recognizer greedy CTC decode in Dart: per-step argmax over 838, collapse
    consecutive duplicates, drop blank@0, map 1..836 -> latin_charset.txt[i-1] and
    837 -> space; confidence = mean max-prob over kept non-blank steps.
[ ] Live text overlay: anchor each decoded string + confidence to its region quad,
    updating as the scene moves; region-count + latency readout on the HUD.
[ ] OPTIONAL local translate step (downstream, Dart, NO ML model): toggle to
    translate the recognized strings on-device. Out of Stage-0 scope; wire as an
    optional module, do not block the core pipeline on it.
[ ] Orientation handling: measure the real buffer WxH on-device and HUD it; do NOT
    assume a landscape buffer (the PyroGuard bug was a SPURIOUS 90-degree rotation
    on an already-upright buffer). Map detected quads back to screen space per the
    measured orientation; round-trip a known quad before trusting the overlay.
[ ] Throttle recognizer re-reads (two model runs per frame = detector + N crops,
    tighter latency budget): only re-run the recognizer on regions that changed /
    persisted, cap crops per frame.
[ ] Tier-A unit tests: recognizer [1,40,838] tensor layout (time-major over 40,
    class over 838); CTC charset off-by-one (blank@0 skipped, i->charset[i-1],
    837->space); coordinate space (736x736 letterbox pixels vs 48x320 padded crop,
    right-pad not decoded as chars); letterbox inverse round-trip; BGR-not-RGB
    channel order; Softmax/Sigmoid already baked (no extra activation); orientation
    round-trip for the measured buffer; de-skewed-crop vs axis-aligned string match.
[ ] A4 hot-path micro-benchmark: mock-tensor timing of detector preprocess +
    post-proc and per-crop recognizer preprocess + CTC decode (Dart pipeline only,
    excludes Melange run).
[ ] Tier-B optimizations: measure each on the Dart hot-path micro-benchmark; avoid
    per-frame double-isolate spawn; reuse buffers; cap/throttle crops.
[ ] Custom domain-identifying launcher icon (street-sign / OCR glyph) from a
    1024x1024 source via flutter_launcher_icons (iOS + Android, remove_alpha_ios).
[ ] Cool, domain-identifying product name as the user-facing display name (iOS
    CFBundleDisplayName, Android android:label, in-app title) — distinct from the
    folder / Melange model names, which stay unchanged.
[ ] iOS signing / release device run: team, NSCameraUsageDescription, iOS 16.6
    min; run release build on a physical iPhone (debug mode hangs on launch). Read
    the SERVED target+apType from the native console; confirm it is not FP32-GPU on
    iOS/macOS 26.3+ (MPSGraph crash trap) — escalate to ZETIC if it is.

Deliverables
- Flutter source under SignTranslate/Flutter/ (loading + live camera screens, HUD,
  MelangeService for both models, detector preprocessor, DB postprocessor +
  region grouping, recognizer preprocessor, CTC decoder, overlay widgets) — PENDING.
- Optional on-device translate module (downstream, no model) — PENDING/optional.
- Model assets PRESENT: export.py, ppocrv5_mobile_det.onnx (+ sample input),
  latin_ppocrv5_mobile_rec.onnx (+ sample input), latin_charset.txt (836 chars),
  model_selection.md, SPEC_stub.md, melange_upload.md.
- Model assets PENDING (GATE 0): registered Melange models
  ajayshah/SceneTextDetector v1 and ajayshah/SceneTextRecognizer v1 with served
  shapes pasted back.
- Diagnostics: this HANDOFF.md (living ticket), plus on-HUD buffer WxH + per-stage
  latency lines (release-build Dart logs do not reach the native console).

References
- App directory: apps/SignTranslate
- Core SDK: ZETIC Melange (zetic_mlange, Flutter FFI) — verify installed version
  before coding; model.create(personalKey, name) / run(List<Tensor>) / close().
- Model A — text DETECTOR: PaddlePaddle/PP-OCRv5_mobile_det (DBNet + MobileNetV3,
  Apache-2.0). float32[1,3,736,736] NCHW BGR, ImageNet normalize ->
  float32[1,1,736,736] prob map [0,1] (Sigmoid baked). DBPostProcess in Dart
  (binarize 0.3, box_thresh 0.6, unclip 1.5). Requested Melange:
  ajayshah/SceneTextDetector v1. modelMode RUN_AUTO.
- Model B — text RECOGNIZER: PaddlePaddle/latin_PP-OCRv5_mobile_rec (SVTR-LCNet
  CTC, Latin, Apache-2.0). float32[1,3,48,320] NCHW BGR, (pixel/255-0.5)/0.5 ->
  [-1,1] -> float32[1,40,838] (Softmax baked). 838 CTC classes: 0=blank,
  1..836=latin_charset.txt, 837=space. Greedy CTC decode in Dart. Requested
  Melange: ajayshah/SceneTextRecognizer v1. modelMode RUN_AUTO.
- Export: paddle2onnx (opset 12) -> onnxslim static input-shapes; single OCR-family
  recipe for both models (see export.py, model_selection.md).
- Frameworks: Flutter, camera plugin, CoreML / Apple Neural Engine (via Melange),
  PaddleOCR / paddle2onnx (export). Optional downstream: local translate (no model).
- Platform: iOS 16.6+, Android minSdk 24. OS trap: FP32-GPU CoreML artifact can
  crash in MPSGraph on iOS/macOS 26.3+ (not client-fixable; ZETIC filters GPU
  server-side). Two model runs per frame — budget for CPU-speed fallback.
- Test device: TBD.
