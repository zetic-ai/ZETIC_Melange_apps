Goal
A real-time, fully on-device sensitive-document redaction demo for Flutter (iOS),
powered by a two-model PP-OCRv5 pipeline through the ZETIC Melange SDK. Point the
camera at an ID / passport / medical form; a DBNet text detector finds text-region
boxes each frame, a CRNN/SVTR CTC recognizer reads budgeted cropped regions, and a
pure-Dart PII heuristic auto-redacts name / DOB / ID-number / MRZ fields live in
the preview before anything is stored or sent. Nothing leaves the device — the
privacy story (fintech ID-scanner + healthcare) is the whole pitch. User-facing
product name: "RedactLens" (display name only; folder, bundle id, and Melange
model names stay unchanged).

Todo List

-- Stage 0 + GATE 0 (done) --
[x] Export DBNet detector (PaddlePaddle/PP-OCRv5_mobile_det_onnx) to static-shape
    ONNX ([1,3,640,640] -> [1,1,640,640], opset 12, onnxslim-folded); static shapes
    verified (no dynamic axis, no Shape op). doc_text_detector.onnx present.
[x] Export CRNN/SVTR CTC recognizer (PaddlePaddle/en_PP-OCRv5_mobile_rec) to
    static-shape ONNX ([1,3,48,320] -> [1,40,438], opset 12, onnxslim-folded);
    static shapes verified. doc_text_recognizer.onnx present.
[x] Emit sample inputs (detector_sample_input.npy, recognizer_sample_input.npy) for
    the Melange dashboard upload.
[x] Emit recognizer charset dict (en_dict.txt, 436 entries) from the model's own
    inference.yml — one char per line, line i -> class i+1 (blank@0, space@437;
    '<' for MRZ is line 80 -> class 81).
[x] Author model_selection.md (top-5 shortlist + winner rationale, both Apache-2.0)
    and SPEC_stub.md (architecture, I/O, pre/post pipeline, traps).
[x] GATE 0 CLEARED — DETECTOR registered on Melange as ajayshah/LiveDocRedact_Detect
    v1 (renamed at registration from the proposed ajayshah/DocTextDetector). Served
    shapes confirmed: input x float32[1,3,640,640], output fetch_name_0
    float32[1,1,640,640]. Benchmarks: NPU med 5.79 ms / GPU med 164 ms / CPU med
    128.6 ms; deployability 98% (FP32 100%). NOTE: dashboard accuracy row shows
    0.00 dB — on-device non-degenerate-heatmap sanity check required (Tier C).
[x] GATE 0 CLEARED — RECOGNIZER registered on Melange as
    ajayshah/LiveDocRedact_Recognize v1 (renamed from the proposed
    ajayshah/DocTextRecognizer). Served shapes confirmed: input x float32[1,3,48,320],
    output fetch_name_0 float32[1,40,438]. Benchmarks: NPU med 1.26 ms / GPU med
    48.9 ms / CPU med 31.9 ms; deployability 98% (FP32 100%).
[x] Finalize SPEC.md (GATE-0 values injected; registered names are binding — the
    old DocText* names in SPEC_stub.md/melange_upload.md are superseded).

-- Build: app skeleton + lifecycle --
[ ] flutter create the app under LiveDocRedact/Flutter/ (bundle id
    com.zeticai.livedocredact); pin zetic_mlange 1.8.1 (PyroGuard-proven surface —
    verify the installed version's actual create()/run() API before coding against
    it), camera, image; PyroGuard-style analysis_options.
[ ] Melange personal key injection (SECRET — never hardcoded/committed):
    const String.fromEnvironment('MELANGE_PERSONAL_KEY') via
    --dart-define=MELANGE_PERSONAL_KEY=..., supplied at build time by the
    human/orchestrator; loading screen shows a clear "key missing" error if empty.
[ ] MelangeService dual-model lifecycle: create BOTH models
    (ajayshah/LiveDocRedact_Detect v1, ajayshah/LiveDocRedact_Recognize v1 — full
    account/name WITH the slash, per CLAUDE.md §5; confirm v1 at first create),
    modelMode RUN_AUTO, per-model download progress -> loading screen; warm BOTH
    with one dummy inference right after load; Tensor.float32List in,
    asFloat32List() out (copy — it is a view over a reused native buffer); close()
    both on dispose.
[ ] Loading screen: dual model download + warm-up progress; camera permission.
[ ] Main screen: live camera preview (BGRA8888 iOS / YUV420 Android, medium
    preset), _busy-guard + drop (never queue) frames, redaction overlay, HUD.

-- Build: pipeline (pure Dart) --
[ ] Long-lived pipeline isolate (NOT per-frame compute() — PyroGuard measured
    ~20 ms/frame double-spawn tax): spawn once, keep the BGR frame resident in the
    isolate; round-trip 1 = frame planes in -> detector input tensor out;
    round-trip 2 = heatmap in -> DB decode + budget selection + K recognizer input
    tensors out. model.run stays on the main isolate (handle bound to it); CTC
    decode + PII classify are trivial and run inline on main.
[ ] Detector preprocessing: source pixel format (BGRA iOS / YUV420 Android) -> BGR
    (drop alpha), letterbox to 640x640 (record scale + pad for the inverse),
    ImageNet normalize (pixel/255 - mean)/std, mean [0.485,0.456,0.406], std
    [0.229,0.224,0.225], applied index-wise to BGR (do NOT reorder to RGB), NCHW
    [1,3,640,640] into a pre-allocated Float32List (single fused pass).
[ ] Detector DB post-processing: read heatmap [1,1,640,640] (Sigmoid baked in —
    apply NO extra activation), binarize thresh 0.3, connected components (two-pass
    labeling), filter by mean heatmap score >= box_thresh 0.6, min-area-rect per
    blob, unclip by ratio 1.5 (offset = area*ratio/perimeter), max_candidates 1000;
    undo the letterbox (exact inverse) -> TextRegion{quad, screen_bbox} in original
    frame space.
[ ] Recognizer per-crop preprocessing: crop the quad from the ORIGINAL BGR frame
    (kept in the pipeline isolate, not the letterboxed tensor), perspective-warp
    upright (text reads left-to-right, H->48), aspect-resize to H=48 then
    right-pad with zeros to W=320 (downscale if wider — pad, never stretch), norm
    (pixel/255 - 0.5)/0.5 -> [-1,1] on BGR, NCHW [1,3,48,320].
[ ] Greedy CTC decoder: load en_dict.txt asset once -> label list [blank](0) +
    436 dict chars (1..436) + ' '(437); per-step argmax over the 438 (LAST) axis
    of [1,40,438], collapse consecutive duplicates THEN drop blank, map to chars;
    mean max-prob as text confidence (output is already Softmax'd — NO extra
    activation); low-confidence results discarded.
[ ] Recognizer BUDGET scheduler (RegionTracker — binding, SPEC mandates it):
    staggered recognition with an IoU-keyed cache PLUS a top-K per-frame cap
    (K default 3 on CPU fallback, tunable; raise when NPU confirmed). Match this
    frame's detector regions to tracked regions by IoU >= 0.5; schedule up to K
    unread/stale regions per frame, prioritizing unread then larger/higher-score;
    cache recognized text + PII class per tracked region so already-read fields
    keep their redaction without re-running; expire regions unseen for M frames.
    Detector boxes + cached PII redactions stay live-overlaid every processed
    frame.
[ ] PII field heuristics (pure Dart on recognized strings + field geometry):
    DOB/date regexes (dd/mm/yyyy, yyyy-mm-dd, dd MMM yyyy, ...) + keyword anchors
    (DOB/Birth/Exp); ID-number patterns (>=6-char alphanumeric-with-digit runs,
    passport/SSN-like formats) + anchors (ID/No./Passport/License/MRN); MRZ lines
    (high '<' density, [A-Z0-9<] charset, ~30-44 chars — '<' is class 81 in the
    dict so it IS recognizable); name fields via keyword anchor (Name/Surname/
    Given) with the value taken from the same field after ':' or the geometrically
    adjacent field. Nothing persisted or transmitted.
[ ] Redaction overlay + HUD: detector boxes live; PII fields covered with solid
    redaction bars (labeled by class; blur variant only if it shows no perf cost);
    "on-device · no cloud" badge; HUD debug line with measured camera buffer WxH +
    preview size + sensor orientation + raw first box (PyroGuard orientation
    lesson — do NOT assume a landscape buffer; the PyroGuard bug was a SPURIOUS
    rotation); per-stage timings (det ms, rec ms/crop, crops this frame, regions
    tracked), detector heatmap min/max/mean (for the 0-dB sanity check), and
    per-class PII counts — all on-screen, since Dart print does not reach a
    release device console.

-- Validation (Tier A) --
[ ] test/ctc_decoder_test.dart — hand-built [1,40,438] logits encoding a known
    string with repeats + blanks -> collapse-repeats-then-drop-blank yields it;
    blank@0; space is the LAST class (437); label-list construction (438 entries,
    labels[1] = first dict line); argmax over the last axis per step (one-hot
    tensor a class-major misread would decode differently); no extra activation
    (confidence equals the raw max value); low-confidence threshold boundary.
[ ] test/recognizer_preprocessor_test.dart — pad-not-stretch round-trip (narrow
    crop -> H=48,W=320 with right-pad columns exactly at the zero-pixel norm value
    -1.0, aspect preserved; wide crop -> downscaled to 320); [-1,1] normalization
    endpoints; BGR channel-order assertion; per-crop deskew (rotated/skewed quad
    with an asymmetric pattern warps upright).
[ ] test/detector_preprocessor_test.dart — letterbox forward scale/pad bookkeeping;
    ImageNet mean/std applied index-wise to BGR (known pixel -> exact expected
    channel values); BGRA->BGR and YUV420->BGR on hand-built buffers; NCHW offsets
    for a single distinctive pixel; full-frame orientation transform round-trips a
    known box for the orientation chosen (0 and 90 degrees).
[ ] test/db_postprocessor_test.dart — hand-built heatmap with one blob -> one quad
    ~ blob rect + unclip, coords in 640-pixel space (not normalized); no extra
    sigmoid (returned region score exactly equals the blob's raw mean); box_thresh
    boundary (0.59 dropped / 0.61 kept); binarize-thresh boundary (0.29 out /
    0.31 in); two adjacent blobs stay two regions (unclip must not merge), one
    contiguous blob stays one region (no word split); full letterbox-inverse
    round-trip of a known original-space box within tolerance.
[ ] test/region_tracker_test.dart — K-cap respected per frame on a mock region
    list; unread regions prioritized; IoU-matched read regions reuse cached text
    and are NOT rescheduled; staggering covers all regions across frames; expiry
    after M unseen frames; cached PII redaction persists while matched.
[ ] test/pii_classifier_test.dart — date/DOB formats, keyword-anchored names,
    ID-number patterns, MRZ '<<' runs, negatives (benign text, short digit runs)
    -> correct classes.
[ ] test/benchmark/hot_path_benchmark.dart (A4) — mock 1280x720 frame + synthetic
    multi-blob heatmap + N~10 regions at K=3 through the full pure-Dart hot path
    (detector preprocess, DB decode, K x crop/warp/pad preprocess, K x CTC decode,
    tracker + PII); median per stage + total over >=100 iterations = the Tier-B
    baseline.
[ ] flutter analyze zero errors/warnings; release device build compiles (A1/A2).

-- Tier B / polish --
[ ] Tier-B optimizations, each with a measured >=0.5% before/after delta on the A4
    benchmark or a logged justification for skipping: long-lived isolate (vs
    compute), pre-allocated buffers, fused preprocess pass, threshold-before-
    geometry in DB decode, repaint overlay only on change, frame drop guard.
[ ] Custom domain-identifying launcher icon (document + redaction-bar/lens glyph)
    from a generated 1024x1024 assets/icon/app_icon.png via flutter_launcher_icons,
    iOS + Android (remove_alpha_ios: true).
[ ] Product name "RedactLens" as user-facing display name only: iOS
    CFBundleDisplayName, Android android:label, in-app MaterialApp title. Bundle
    id, folder (LiveDocRedact), and Melange names (ajayshah/LiveDocRedact_Detect,
    ajayshah/LiveDocRedact_Recognize) stay unchanged.

-- GATE 3 / device (human-gated) --
[ ] iOS signing / release-build device run: team, NSCameraUsageDescription,
    iOS 16.6 min, vendored ZeticMLange.xcframework; physical iPhone, RELEASE build
    (simulator is a dead end: device-only slice + no camera; debug hangs).
[ ] Tier-C checklist at handoff: read the SERVED artifact (runtimeApType +
    target/precision) for BOTH models from the native console
    (xcrun devicectl device process launch --console ...); confirm not GPU on
    iOS/macOS 26.3+ (MPSGraph trap); verify the detector heatmap is NON-DEGENERATE
    on-device (HUD min/max/mean vary, text lights up — dashboard accuracy row read
    0.00 dB); re-tune K/stagger to the served backend; first-launch DOUBLE model
    download on conference Wi-Fi (pre-warm); multi-cold-start acceptance; personal
    key embedded in client (build-time define, not in repo).
[ ] Android run verification once iOS is stable.

Deliverables
- Flutter source under LiveDocRedact/Flutter/ (loading + main screens, dual-model
  MelangeService, pipeline isolate, detector preprocessor, DB post-processor,
  recognizer crop/warp preprocessor, CTC decoder, region tracker / budget
  scheduler, PII heuristics, redaction overlay + HUD widgets, Tier-A tests,
  hot-path benchmark).
- Model assets (present): export.py, doc_text_detector.onnx, doc_text_recognizer.onnx,
  detector_sample_input.npy, recognizer_sample_input.npy, en_dict.txt (436 entries).
- Melange models (REGISTERED, GATE 0 cleared Jul 2 2026):
  ajayshah/LiveDocRedact_Detect v1 (x[1,3,640,640] -> fetch_name_0[1,1,640,640])
  and ajayshah/LiveDocRedact_Recognize v1 (x[1,3,48,320] -> fetch_name_0[1,40,438]).
- iOS config (planned): signing team, Info.plist camera usage, Podfile (iOS 16.6,
  vendored ZeticMLange.xcframework).
- Diagnostics: this HANDOFF.md (living plan-of-record), on-screen latency /
  regions-per-frame / heatmap-stats HUD, native device-console (devicectl)
  workflow.

References
- App directory: apps/LiveDocRedact
- Core SDK: ZETIC Melange (zetic_mlange 1.8.1 planned pin — PyroGuard-proven;
  verify installed API surface before coding)
- Model 1 (detector): PP-OCRv5 mobile DBNet — PaddlePaddle/PP-OCRv5_mobile_det_onnx
  (Apache-2.0), registered ajayshah/LiveDocRedact_Detect v1, input
  float32[1,3,640,640] BGR ImageNet-norm, output float32[1,1,640,640] probability
  heatmap (Sigmoid baked in). NPU med 5.79 ms / CPU med 128.6 ms.
- Model 2 (recognizer): en_PP-OCRv5 mobile CRNN/SVTR CTC —
  PaddlePaddle/en_PP-OCRv5_mobile_rec (Apache-2.0), registered
  ajayshah/LiveDocRedact_Recognize v1, input float32[1,3,48,320] BGR [-1,1] (fixed
  width 320), output float32[1,40,438] (40 steps x 438 classes: blank@0 +
  en_dict 1..436 + space@437). NPU med 1.26 ms / CPU med 31.9 ms per crop.
- Export: paddle2onnx / HF pre-exported ONNX, onnxslim-fold, opset 12 (see export.py).
- Reference implementation: apps/FireDetectionYOLO/Flutter (PyroGuard) — structure,
  MelangeService lifecycle, orientation HUD lesson, isolate-cost lesson.
- Frameworks: Flutter, camera plugin, CoreML / Apple Neural Engine (via Melange).
- Test device: TBD (likely the PyroGuard iPhone 15, iOS 26.x — confirm at GATE 3).
