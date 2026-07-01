Goal
A fully offline, on-device receipt/invoice OCR demo for Flutter (iOS), for
field-sales and enterprise expense workflows where connectivity is poor and
receipt financials must not touch a third-party cloud OCR. The user uploads a
still photo of a receipt; a two-model PP-OCRv5 pipeline through the ZETIC Melange
SDK (DB detector -> SVTR/CTC recognizer) localizes and reads each text line, and
a pure-Dart structuring layer clusters the boxes into rows and pulls line-items,
total, merchant, and date into structured fields. This is the "any model + real
post-processing" proof point: the models emit text + boxes; Dart does the rest.

Todo List
[x] Select models and write model_selection.md (PP-OCRv5 mobile det + en rec,
    Apache-2.0; two-model justification: a single recognizer cannot localize the
    many receipt text lines, and end-to-end OCRs are autoregressive/dynamic-shape
    and not Melange-fit).
[x] Author SPEC_STUB.md (full pre/post-processing pipelines, I/O shapes, text+box
    output contract, structuring heuristics, Tier-A validation focus).
[x] Export DETECTOR to ONNX (ppocrv5_mobile_det.onnx, ~4.5 MB, opset 12, static
    [1,3,960,640] -> [1,1,960,640], Sigmoid baked in) via paddle2onnx -> onnxslim.
[x] Export RECOGNIZER to ONNX (en_ppocrv5_mobile_rec.onnx, ~7.5 MB, opset 12,
    static [1,3,48,320] -> [1,40,438], Softmax baked in) via paddle2onnx -> onnxslim.
[x] Generate sample inputs for both models (dashboard upload + forward-pass check).
[x] Ship the recognizer charset: rec_charset.json (438-entry index-ordered table)
    and rec_dict.txt (raw 436-char dict); blank=0, dict=1..436, space=437.
[ ] [BLOCKED - human/dashboard, GATE 0] Register BOTH models on Melange and paste
    back served name/version + served I/O shapes: ajayshah/ReceiptScannerDet v1
    (in [1,3,960,640] -> out [1,1,960,640]) and ajayshah/ReceiptScannerRec v1
    (in [1,3,48,320] -> out [1,40,438]). App is blocked until both are READY.
[ ] Create core Flutter structure: loading screen (dual-model download + warm-up),
    upload / pick-image screen (gallery + camera-capture-to-file), result view
    (receipt with overlaid line boxes + structured results panel).
[ ] MelangeService lifecycle for BOTH models: create(personalKey, name) ->
    Tensor.float32List -> run -> asFloat32List -> close; warm-up each model once at
    load; budget N sequential REC calls per receipt (static batch=1, no batching).
[ ] DET preprocessing: decode upload with EXIF orientation -> letterbox to 960x640
    (record scale + pad offsets) -> BGR channel order -> /255 -> ImageNet
    mean [0.485,0.456,0.406] std [0.229,0.224,0.225] -> NCHW [1,3,960,640].
[ ] DET post-processing (DB, pure Dart): read [1,1,960,640] prob map -> binarize at
    0.3 -> connected-components/contours -> min-area rotated boxes -> drop box_thresh
    < 0.6 -> unclip x1.5 -> invert letterbox to image space (round-trippable).
[ ] REC line-crop pipeline: crop + perspective-deskew each box from the ORIGINAL
    image -> resize to height 48 keep-aspect -> right-pad to width 320 (scale down if
    > 320) -> BGR -> /255 then (x-0.5)/0.5 -> [-1,1] -> NCHW [1,3,48,320].
[ ] REC greedy CTC decode (pure Dart): read [1,40,438] as TIME-MAJOR (step t occupies
    t*438 .. t*438+437) -> argmax per step -> collapse consecutive repeats -> drop
    blank(0) -> map via rec_charset.json (space=437). A transposed read yields
    plausible-but-wrong text. Confidence = mean max-prob over kept non-blank steps.
[ ] Structured extraction in Dart (FIRST-CLASS deliverable, NOT a model): cluster
    RecognizedLines into rows by y-overlap, sort L->R, then regex + layout heuristics
    for line-items (description + trailing price \d+[.,]\d{2}), total
    (TOTAL|AMOUNT DUE|BALANCE|GRAND TOTAL, largest/last currency value, guard
    SUBTOTAL/TAX), merchant (top-most large text block), date (common formats).
[ ] Tier-A unit tests: REC time-major decode (hand-built matrix, one known argmax per
    step); CTC off-by-one / charset mapping ("aa[blank]b" -> "ab", index->space);
    letterbox inverse round-trip (DET, within tolerance); coordinate space
    (pixel not normalized); BGR-vs-RGB channel order; per-model normalization
    (ImageNet vs [-1,1]); DET threshold/box_thresh/unclip boundaries; REC resize+pad
    (padded region decodes to blank/space); EXIF orientation mapping.
[ ] A4 hot-path micro-benchmark (mock tensors): DET preprocess + Dart DB post-proc +
    one REC preprocess + CTC decode; capture Dart-side cost separate from run().
[ ] Tier-B optimizations with measured deltas on the Dart hot-path (or a justified
    skip): cap/queue REC line count, avoid redundant isolate spawns, tighten
    per-line crop/normalize allocations.
[ ] Custom launcher icon: domain-identifying receipt glyph, 1024x1024 source at
    Flutter/assets/icon/app_icon.png, generated for iOS + Android via
    flutter_launcher_icons (remove_alpha_ios: true). Default Flutter icon not allowed.
[ ] Cool product name as user-facing display name (suggested "Expensa" / "ReceiptIQ"):
    iOS CFBundleDisplayName, Android android:label, in-app MaterialApp title / app-bar
    / loading text. Bundle id, folder, and Melange model names stay ReceiptScanner*.
[ ] iOS signing/release config (team, NSPhotoLibraryUsageDescription /
    NSCameraUsageDescription, iOS 16.6 min, vendored ZeticMLange.xcframework) and run
    on a physical device in release mode; read the SERVED target+apType from the native
    console and confirm it is not a GPU artifact on iOS/macOS 26.3+ for BOTH models.
[ ] Android run verification once iOS is stable (minSdk 24).

Deliverables
- Planned Flutter source under ReceiptScanner/Flutter/ (loading/upload/result
  screens, MelangeService for both models, DET + REC preprocessors, DB detector
  post-processor, greedy CTC decoder, RecognizedLine model, overlay + results panel,
  latency HUD).
- Structured-extraction module (pure Dart, worker-owned): row-clustering + regex/
  layout heuristics for merchant / date / line-items / total. The proof-point.
- Model assets: export.py (paddle2onnx -> onnxslim recipe), ppocrv5_mobile_det.onnx,
  en_ppocrv5_mobile_rec.onnx, sample inputs, rec_charset.json + rec_dict.txt
  (present) + registered Melange models ajayshah/ReceiptScannerDet v1 and
  ajayshah/ReceiptScannerRec v1 (pending GATE 0).
- Diagnostics: this HANDOFF.md (living ticket), Tier-A test battery, hot-path
  micro-benchmark, on-screen/HUD diagnostics (served artifact, per-stage det + N*rec
  + Dart latency) since Dart print does not reach the release device console.

References
- App directory: apps/ReceiptScanner
- Core SDK: ZETIC Melange (zetic_mlange, Flutter FFI; confirm installed version
  before coding; create/run/close API, Tensor.float32List / asFloat32List)
- Model A (detector): PaddlePaddle/PP-OCRv5_mobile_det (Apache-2.0), PP-OCRv5 mobile
  DB text detector, in float32[1,3,960,640] NCHW BGR ImageNet-norm ->
  out float32[1,1,960,640] prob map (Sigmoid baked in); DB box extraction is Dart.
- Model B (recognizer): PaddlePaddle/en_PP-OCRv5_mobile_rec (Apache-2.0), PP-OCRv5
  mobile English SVTR-LCNet + CTC head, in float32[1,3,48,320] NCHW BGR [-1,1] ->
  out float32[1,40,438] softmaxed CTC posteriors (time-major); greedy decode is Dart.
  Charset: 438 classes, blank=0, dict 1..436, space=437 (rec_charset.json / rec_dict.txt).
- Export: paddle2onnx --opset_version 12 then onnxslim --input-shapes (static-shape
  pin + constant-fold; onnxsim segfaulted on arm64). Both verified static, opset 12.
- Platform: iOS 16.6+, Android minSdk 24. Trap: FP32-GPU CoreML artifact can crash in
  MPSGraph on iOS/macOS 26.3+ (not client-fixable; ZETIC filters GPU server-side) —
  applies to BOTH models; realistic non-crashing fallback is TFLITE_FP16/CPU.
- Test device: TBD.
