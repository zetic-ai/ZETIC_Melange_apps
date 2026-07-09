import 'dart:typed_data';

/// Number of quantile rows the model emits (levels 0.1 .. 0.9).
const int kNumQuantiles = 9;

/// Forecast horizon in samples.
const int kHorizon = 64;

/// Index of the median (0.5) quantile row.
const int kMedianRow = 4;

/// One decoded model output: a 64-step quantile forecast, anchored on the
/// global sample index of the FIRST predicted step.
///
/// The raw tensor is float32[1, 9, 64], quantile-major: flat index = q*64 + t.
/// Values are already in original data units (de-normalization is in-graph).
class Forecast {
  Forecast({required this.anchorIndex, required Float32List raw})
      : assert(raw.length == kNumQuantiles * kHorizon,
            'expected ${kNumQuantiles * kHorizon} floats, got ${raw.length}'),
        _raw = raw;

  /// Global sample index that horizon step 0 predicts. A window ending at
  /// global index `w` produces `anchorIndex = w + 1`.
  final int anchorIndex;

  final Float32List _raw;

  /// Value for quantile row [q] (0..8) at horizon step [t] (0..63).
  double at(int q, int t) => _raw[q * kHorizon + t];

  double q10(int t) => at(0, t);
  double q30(int t) => at(2, t);
  double median(int t) => at(kMedianRow, t);
  double q70(int t) => at(6, t);
  double q90(int t) => at(8, t);

  /// Horizon step for a global sample index, or -1 if outside 0..63.
  int stepFor(int globalIndex) {
    final t = globalIndex - anchorIndex;
    return (t >= 0 && t < kHorizon) ? t : -1;
  }
}

/// A flagged anomaly (post-debounce) for the event list.
class AnomalyEvent {
  const AnomalyEvent({
    required this.globalIndex,
    required this.wallClock,
    required this.value,
    required this.score,
  });

  final int globalIndex;
  final DateTime wallClock;
  final double value;
  final double score;
}
