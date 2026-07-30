import '../config.dart';
import 'region_tracker.dart';

/// Frame/budget scheduler implementing the SPEC-mandated behaviors:
///
/// 1. top-K recognition per frame ([selectForRecognition]),
/// 3. adaptive detection cadence from the measured detection-pass EMA
///    ([shouldRunDetection]),
/// 4. `_busy` frame dropping — frames are dropped, never queued
///    ([tryBeginPass]/[endPass]),
/// 5. HUD latency readouts (the recorded stats exposed here).
///
/// (Behavior 2, the IoU cache, lives in [RegionTracker].)
///
/// The clock is injectable so cadence logic is testable with a fake clock.
class FrameScheduler {
  FrameScheduler({
    int Function()? clock,
    this.topK = kTopK,
    this.dutyTarget = kDetectionDutyTarget,
    this.frameBudgetMs = kFrameBudgetMs,
    this.maxIntervalMs = kMaxDetectionIntervalMs,
  }) : _nowMs = clock ?? (() => DateTime.now().millisecondsSinceEpoch);

  final int Function() _nowMs;
  final int topK;
  final double dutyTarget;
  final int frameBudgetMs;
  final int maxIntervalMs;

  bool _busy = false;
  bool get isBusy => _busy;

  int _lastDetectionEndMs = 0;
  bool _hasDetected = false;

  double _emaDetPassMs = 0;
  double _emaDetModelMs = 0;
  double _emaRecModelMs = 0;
  int _droppedFrames = 0;

  /// EMA of the FULL detection pass (preprocess + run + postprocess) — this
  /// drives the cadence.
  double get emaDetectionPassMs => _emaDetPassMs;

  /// EMA of the detector model.run alone (HUD).
  double get emaDetectorModelMs => _emaDetModelMs;

  /// EMA of one recognizer model.run (HUD, per crop).
  double get emaRecognizerModelMs => _emaRecModelMs;

  /// Frames dropped by the busy guard since start (HUD).
  int get droppedFrames => _droppedFrames;

  // --- Behavior 4: busy guard / frame dropping. -----------------------------

  /// Claims the pipeline for one camera frame. Returns false — and counts a
  /// dropped frame — while a pass is already in flight. Frames are NEVER
  /// queued.
  bool tryBeginPass() {
    if (_busy) {
      _droppedFrames++;
      return false;
    }
    _busy = true;
    return true;
  }

  void endPass() {
    _busy = false;
  }

  // --- Behavior 3: adaptive detection cadence. ------------------------------

  /// Whether this frame should run the detector.
  ///
  /// Fast passes (<= one frame budget, i.e. the NPU case) detect every frame.
  /// Slow passes (CPU fallback, ~169 ms+) are spaced so detection consumes at
  /// most [dutyTarget] of wall time: wait `pass * (1-duty)/duty` after each
  /// pass ends (duty 0.5 -> wait one pass-length), capped at [maxIntervalMs].
  /// Cached overlays keep tracking between detection frames via the tracker.
  bool shouldRunDetection() {
    if (!_hasDetected) return true;
    if (_emaDetPassMs <= frameBudgetMs) return true;
    var waitMs = _emaDetPassMs * (1 - dutyTarget) / dutyTarget;
    if (waitMs > maxIntervalMs) waitMs = maxIntervalMs.toDouble();
    return _nowMs() - _lastDetectionEndMs >= waitMs;
  }

  /// Records a completed detection pass ([passMs] = full pipeline,
  /// [modelMs] = detector model.run alone).
  void recordDetectionPass({required int passMs, required int modelMs}) {
    _emaDetPassMs = _ema(_emaDetPassMs, passMs);
    _emaDetModelMs = _ema(_emaDetModelMs, modelMs);
    _lastDetectionEndMs = _nowMs();
    _hasDetected = true;
  }

  /// Records one recognizer model.run.
  void recordRecognition(int modelMs) {
    _emaRecModelMs = _ema(_emaRecModelMs, modelMs);
  }

  double _ema(double current, int sample) =>
      current == 0 ? sample.toDouble() : current * 0.7 + sample * 0.3;

  // --- Behavior 1: top-K crop selection. ------------------------------------

  /// Prioritizes recognition candidates: cache-misses only (hits never reach
  /// here), largest area first (nearest/most prominent signs), capped at
  /// [limit] (defaults to [topK]) per frame. The remainder is the caller's
  /// staggered-recognition queue for subsequent frames.
  List<TrackedRegion> selectForRecognition(
    List<TrackedRegion> misses, {
    int? limit,
  }) {
    final sorted = [...misses]
      ..sort((a, b) => b.quad.area.compareTo(a.quad.area));
    final k = limit ?? topK;
    return sorted.length <= k ? sorted : sorted.sublist(0, k);
  }
}
