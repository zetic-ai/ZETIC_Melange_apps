Goal
A real-time, fully on-device sensitive-document redaction demo for Flutter (iOS),
powered by a two-model PP-OCRv5 pipeline through the ZETIC Melange SDK. Point the
camera at an ID / passport / medical form; a DBNet text detector finds text-region
boxes each frame, a CRNN/SVTR CTC recognizer reads each cropped region, and a
pure-Dart PII heuristic auto-blurs/boxes name / DOB / ID-number / MRZ fields live
in the preview before anything is stored or sent. Nothing leaves the device — the
privacy story (fintech ID-scanner + healthcare) is the whole pitch.

Todo List
[x] Export DBNet detector (PaddlePaddle/PP-OCRv5_mobile_det_onnx) to static-shape
    ONNX ([1,3,640,640] -> [1,1,640,640], opset 12, onnxslim-folded); static shapes
    verified (no dynamic axis, no Shape op). doc_text_detector.onnx present.
[x] Export CRNN/SVTR CTC recognizer (PaddlePaddle/en_PP-OCRv5_mobile_rec) to
    static-shape ONNX ([1,3,48,320] -> [1,40,438], opset 12, onnxslim-folded);
    static shapes verified. doc_text_recognizer.onnx present.
[x] Emit sample inputs (detector_sample_input.npy, recognizer_sample_input.npy) for
    the Melange dashboard upload.
[x] Emit recognizer charset dict (en_dict.txt, 436 entries) from the model's own
    inference.yml — one char per line, line i -> class i+1.
[x] Author model_selection.md (top-5 shortlist + winner rationale, both Apache-2.0)
    and SPEC_stub.md (architecture, I/O, pre/post pipeline, traps).
[ ] [BLOCKED - human/dashboard, GATE 0] Register the DETECTOR on Melange
    (requested ajayshah/DocTextDetector, v1): human drags doc_text_detector.onnx +
    detector_sample_input.npy into the dashboard, waits for READY, pastes back the
    registered name + version and served input/output shapes. App blocked until then.
[ ] [BLOCKED - human/dashboard, GATE 0] Register the RECOGNIZER on Melange
    (requested ajayshah/DocTextRecognizer, v1): human drags doc_text_recognizer.onnx +
    recognizer_sample_input.npy into the dashboard, waits for READY, pastes back the
    registered name + version and served input/output shapes. App blocked until then.
[ ] Create core Flutter structure: loading screen (dual model download + warm-up
    progress) and live camera screen with detected-box overlay, PII blur/box layer,
    "on-device . no cloud" badge, and a latency / regions-per-frame HUD (on-screen,
    since Dart print won't reach a release device console).
[ ] MelangeService lifecycle for BOTH models: create -> Tensor.float32List -> run ->
    close; warm each with a dummy inference right after load; _busy-guard / throttle
    frames (hot path is 1 detector + N recognizer inferences per frame).
[ ] Detector preprocessing: source pixel format (BGRA iOS / YUV420 Android) -> BGR,
    letterbox to 640x640 (record scale + pad for the inverse), ImageNet normalize
    ((pixel/255 - mean)/std, mean [0.485,0.456,0.406], std [0.229,0.224,0.225])
    applied index-wise to BGR (do NOT reorder to RGB), NCHW [1,3,640,640] Float32List.
[ ] Detector DB post-processing in Dart: read heatmap [1,1,640,640] (Sigmoid baked
    in - apply NO extra sigmoid), binarize thresh ~0.3, connected-components/contours,
    filter by mean score >= box_thresh ~0.6, unclip ~1.5, min-area-rect quad boxes.
[ ] Undo the letterbox (exact inverse of detector pre-step 2) to map each quad back to
    screen/original space; emit TextRegion{quad, screen_bbox}.
[ ] Text-region grouping into reading lines (Dart), without merging adjacent fields or
    splitting a word.
[ ] Recognizer per-crop preprocessing: crop the quad from the ORIGINAL frame,
    deskew/perspective-warp upright (text reads left-to-right, H->48), aspect-resize to
    H=48 then right-pad with zeros to W=320 (downscale if wider), BGR norm
    ((pixel/255 - 0.5)/0.5) -> [-1,1], NCHW [1,3,48,320] Float32List.
[ ] Recognizer greedy CTC decode in Dart: per-step argmax over the 438 (last) axis of
    [1,40,438], collapse consecutive duplicates, drop blank (class 0); map indices via
    label list ['blank'] + en_dict.txt(1..436) + [' '](437) -> the region's string.
[ ] PII field heuristics (pure-Dart regex + keyword anchors): classify each read field
    as name / DOB / ID-number / MRZ (`<<` runs) / other; live blur/box the PII
    screen_bbox in the preview; persist and transmit nothing.
[ ] Orientation handling: measure the real buffer WxH on-device; do NOT assume a
    landscape buffer (the PyroGuard bug was a SPURIOUS 90-degree rotation, not a
    missing one) - surface prev/buf/sensor on the HUD to confirm.
[ ] Tier-A test - CTC decode semantics (#1 silent-wrong): hand-built [1,40,438] logits
    encoding a known string with repeats + blanks -> assert collapse-repeats-then-
    drop-blank yields the string; blank@0, space is the LAST class (437).
[ ] Tier-A test - recognizer output layout: argmax is over the 438 axis per step, NOT
    across steps; assert against a hand-built one-hot-per-step tensor.
[ ] Tier-A test - fixed-width padding round-trip: a crop narrower than 320 is
    right-padded with zeros (not stretched); assert aspect-resize + pad -> H=48,W=320
    and padding emits no spurious characters.
[ ] Tier-A test - detector heatmap decode + letterbox inverse: round-trip a known box
    (forward letterbox -> DB unclip -> inverse letterbox) back to original within
    tolerance; coords are in 640x640 pixel space, not normalized.
[ ] Tier-A test - activation semantics: detector heatmap is Sigmoid-baked and
    recognizer head is Softmax'd; assert NO extra activation is applied in Dart.
[ ] Tier-A test - channel order (BGR vs RGB): assert the preprocessor keeps BGR for
    both models (a silent R/B swap degrades accuracy without throwing).
[ ] Tier-A test - orientation round-trip (twice): (i) full-frame buffer orientation
    into the detector; (ii) per-crop deskew makes a sideways/upside-down quad upright.
[ ] Tier-A test - text-region geometry: two adjacent fields must not merge (unclip too
    large) nor split a word; on a hand-built heatmap. Plus threshold-boundary tests
    (box_thresh just-below dropped / just-above kept).
[ ] A4 hot-path micro-benchmark: mock-tensor timing of preprocess / detector run /
    DB decode / N x (crop+recognizer run+CTC decode) on a representative region count.
[ ] Tier-B optimizations: each applied with its measured delta on the Dart hot-path
    micro-benchmark, or a justification for skipping (e.g. isolate/frame-throttle,
    reuse buffers, cap regions-per-frame).
[ ] Custom domain-identifying launcher icon (document/redaction glyph) from a
    1024x1024 source via flutter_launcher_icons, iOS + Android (remove_alpha_ios: true).
[ ] Set a cool, domain-identifying product name (suggested "RedactLens") as the
    user-facing display name only: iOS CFBundleDisplayName, Android android:label,
    in-app title. Bundle id, folder (LiveDocRedact), and Melange names (ajayshah/
    DocText*) stay unchanged.
[ ] iOS signing / release-build device run: team, NSCameraUsageDescription, iOS 16.6
    min, vendored ZeticMLange.xcframework; run on a physical iPhone (release build -
    simulator is a dead end, no device-only slice + no camera). Confirm the SERVED
    artifact is not GPU on iOS/macOS 26.3+ via the device console (MPSGraph trap).
[ ] Android run verification once iOS is stable.

Deliverables
- Flutter source under LiveDocRedact/Flutter/ (loading + live camera screens,
  MelangeService for both models, preprocessor, DB detector post-processor,
  recognizer CTC decoder, region-grouping/crop-warp helper, PII heuristics,
  overlay/blur + HUD widgets, Tier-A tests, hot-path benchmark).
- Model assets (present): export.py, doc_text_detector.onnx, doc_text_recognizer.onnx,
  detector_sample_input.npy, recognizer_sample_input.npy, en_dict.txt (436 entries).
- Melange models (PENDING GATE-0 registration): ajayshah/DocTextDetector v1 and
  ajayshah/DocTextRecognizer v1 - names/versions/served shapes to be pasted back.
- iOS config (planned): signing team, Info.plist camera usage, Podfile (iOS 16.6,
  vendored ZeticMLange.xcframework).
- Diagnostics: this HANDOFF.md (living plan-of-record), on-screen latency /
  regions-per-frame HUD, native device-console (devicectl) workflow.

References
- App directory: apps/LiveDocRedact
- Core SDK: ZETIC Melange (zetic_mlange, Flutter FFI)
- Model 1 (detector): PP-OCRv5 mobile DBNet - PaddlePaddle/PP-OCRv5_mobile_det_onnx
  (Apache-2.0), input float32[1,3,640,640] BGR ImageNet-norm, output float32[1,1,640,640]
  probability heatmap (Sigmoid baked in). ~4.75 MB.
- Model 2 (recognizer): en_PP-OCRv5 mobile CRNN/SVTR CTC -
  PaddlePaddle/en_PP-OCRv5_mobile_rec (Apache-2.0), input float32[1,3,48,320] BGR
  [-1,1] (fixed width 320), output float32[1,40,438] (40 steps x 438 classes: blank@0
  + en_dict 1..436 + space@437). ~7.8 MB.
- Export: paddle2onnx / HF pre-exported ONNX, onnxslim-fold, opset 12 (see export.py).
- Frameworks: Flutter, camera plugin, CoreML / Apple Neural Engine (via Melange).
- Test device: TBD.
