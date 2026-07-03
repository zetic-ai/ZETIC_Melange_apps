/// GlyphGo build-time configuration and tunable pipeline constants.
///
/// All SPEC-binding values live here so tests and services share one source
/// of truth (736 letterbox, 838 CTC classes, DB thresholds 0.3/0.6/1.5, ...).
library;

/// ZETIC Melange personal key, injected at build time — never committed:
///
/// ```sh
/// flutter build ios --release --dart-define=MELANGE_PERSONAL_KEY=<your key>
/// ```
///
/// When empty, the loading screen shows an explanatory error instead of
/// attempting a model load.
const String kMelangePersonalKey = String.fromEnvironment(
  'MELANGE_PERSONAL_KEY',
);

/// Fully-qualified Melange model names — the `ajayshah/` account prefix WITH
/// the slash is required (a bare project name throws MlangeException(3)
/// on-device at load). The dashboard's "ZETIC |" header is a display prefix,
/// not the account.
const String kDetectorModelName = 'ajayshah/SignTranslate_Detect';
const String kRecognizerModelName = 'ajayshah/SignTranslate_Rec';

/// First upload (2026-07-02) — version 1 assumed; confirmed at first create.
const int kModelVersion = 1;

// ---------------------------------------------------------------------------
// Detector — DBNet (PP-OCRv5 mobile det), float32[1,3,736,736] BGR NCHW.
// ---------------------------------------------------------------------------

/// Detector letterbox size. 736, NOT 640 (divisible by 32, DBNet requirement).
const int kDetInputSize = 736;

/// ImageNet per-channel normalization applied after /255, in the exported
/// channel order 0,1,2 (BGR — PaddleOCR keeps cv2 BGR; do NOT swap to RGB).
const List<double> kDetMean = [0.485, 0.456, 0.406];
const List<double> kDetStd = [0.229, 0.224, 0.225];

/// DB post-processing parameters (SPEC-binding).
const double kDbProbThreshold = 0.3;
const double kDbBoxThreshold = 0.6;
const double kDbUnclipRatio = 1.5;

/// Boxes whose short side (before unclip) is smaller than this are noise.
const double kDbMinBoxSize = 3.0;

// ---------------------------------------------------------------------------
// Recognizer — SVTR-LCNet CTC (latin PP-OCRv5 mobile rec),
// float32[1,3,48,320] BGR NCHW in [-1,1] -> float32[1,40,838] (Softmax baked).
// ---------------------------------------------------------------------------

const int kRecHeight = 48;
const int kRecWidth = 320;
const int kRecTimeSteps = 40;

/// 838 CTC classes — NOT 438: blank@0 + latin_charset.txt lines 1..836 +
/// space@837.
const int kRecNumClasses = 838;

/// Tall deskewed crops (h >= 1.5*w) are rotated 90° CCW before recognition
/// (PaddleOCR get_rotate_crop_image parity — vertical-ish signage).
const double kVerticalCropRatio = 1.5;

// ---------------------------------------------------------------------------
// Frame scheduling (all five behaviors SPEC-mandated).
// ---------------------------------------------------------------------------

/// Recognize at most K regions per processed frame.
const int kTopK = 3;

/// A detected quad matching a cached region with bbox IoU >= this reuses the
/// cached string without re-running the recognizer.
const double kIouCacheThreshold = 0.5;

/// Cached regions unmatched for this many detection cycles are evicted.
const int kEvictAfterMissedCycles = 8;

/// Adaptive detection cadence: when a detection pass costs more than one
/// frame budget, wait `passMs * (1 - duty) / duty` after each pass so
/// detection consumes at most this fraction of wall time.
const double kDetectionDutyTarget = 0.5;

/// One camera frame at ~30 fps; detection passes cheaper than this run
/// every frame.
const int kFrameBudgetMs = 33;

/// Never wait longer than this between detection passes.
const int kMaxDetectionIntervalMs = 1500;
