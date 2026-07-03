import 'dart:math' as math;
import 'dart:typed_data';

import '../models/forecast.dart';

/// Floor for the q10-q90 band width so a degenerate (zero-width) band never
/// produces NaN/Inf scores.
const double kIqrFloor = 1e-6;

/// Default anomaly threshold (Stage-0 measured: 1.0 gives 4/4 NAB failure
/// windows at 1.07% raw FP before debounce).
const double kDefaultThreshold = 1.0;

/// Consecutive exceedances required before a flag is raised.
const int kDebounceCount = 2;

/// Decodes the raw float32[1,9,64] model output (quantile-major, flat index
/// q*64+t) into a [Forecast] anchored at [anchorIndex].
///
/// [raw] is copied: the SDK's asFloat32List() is a view over a reused native
/// buffer, so the caller may pass it straight in.
Forecast decodeForecast(Float32List raw, {required int anchorIndex}) {
  return Forecast(anchorIndex: anchorIndex, raw: Float32List.fromList(raw));
}

/// Band-exceedance anomaly score for actual value [x] at horizon step [t] of
/// [f]:  max(0, (x - q90)/iqr, (q10 - x)/iqr), iqr = max(q90 - q10, floor).
/// 0 inside the band; 1.0 means one full band-width outside it.
double anomalyScore(Forecast f, int t, double x) {
  final q10 = f.q10(t);
  final q90 = f.q90(t);
  final iqr = math.max(q90 - q10, kIqrFloor);
  final above = (x - q90) / iqr;
  final below = (q10 - x) / iqr;
  return math.max(0.0, math.max(above, below));
}

/// Debounce state machine: raises a flag only after [debounceCount]
/// CONSECUTIVE samples score >= [threshold]. A single outlier sample (or an
/// exceedance broken by a normal sample) never flags.
class AnomalyDetector {
  AnomalyDetector({
    this.threshold = kDefaultThreshold,
    this.debounceCount = kDebounceCount,
  });

  /// Exposed as a UI slider (0.5 - 3.0).
  double threshold;
  final int debounceCount;

  int _streak = 0;

  int get streak => _streak;

  /// Feeds one scored sample; returns true when the debounce fires (i.e. this
  /// sample is the [debounceCount]-th consecutive exceedance or later).
  bool onScore(double score) {
    if (score >= threshold) {
      _streak++;
    } else {
      _streak = 0;
    }
    return _streak >= debounceCount;
  }

  void reset() => _streak = 0;
}
