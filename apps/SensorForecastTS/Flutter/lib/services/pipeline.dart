import 'dart:typed_data';

import '../models/forecast.dart';
import 'postprocessor.dart';
import 'preprocessor.dart';

/// Result of feeding one sample through the pipeline.
class PipelineResult {
  const PipelineResult({
    required this.globalIndex,
    required this.score,
    required this.flagged,
    required this.needForecast,
  });

  final int globalIndex;

  /// Band-exceedance score vs the latest forecast, or null when no forecast
  /// covers this sample yet.
  final double? score;

  /// True when the debounce fired on this sample (>= 2 consecutive
  /// exceedances at >= threshold).
  final bool flagged;

  /// True when the caller should run a model forecast now (window full AND
  /// re-forecast cadence reached, or no forecast exists yet).
  final bool needForecast;
}

/// Pure (SDK-free, timer-free) orchestration of window + forecast + scoring,
/// so the alignment/debounce logic is unit-testable end to end.
///
/// Ordering per sample (SPEC post-processing):
///   1. score the incoming sample against the CURRENT forecast at horizon
///      step (globalIndex - anchorIndex),
///   2. push it into the window,
///   3. report whether a re-forecast is due (every [reforecastEvery] samples).
/// After the caller runs the model, [applyForecast] anchors the new forecast
/// at the next incoming sample's global index.
class ForecastPipeline {
  ForecastPipeline({
    this.reforecastEvery = 8,
    AnomalyDetector? detector,
    SampleWindow? window,
  })  : detector = detector ?? AnomalyDetector(),
        window = window ?? SampleWindow();

  final int reforecastEvery;
  final AnomalyDetector detector;
  final SampleWindow window;

  Forecast? forecast;
  int _globalIndex = 0; // index the NEXT pushed sample will occupy
  int _sinceForecast = 0;

  int get globalIndex => _globalIndex;

  PipelineResult push(double x) {
    final idx = _globalIndex;
    double? score;
    var flagged = false;

    final f = forecast;
    if (f != null) {
      final t = f.stepFor(idx);
      if (t >= 0) {
        score = anomalyScore(f, t, x);
        flagged = detector.onScore(score);
      }
    }

    window.push(x);
    _globalIndex++;
    _sinceForecast++;

    final needForecast = window.isFull &&
        (f == null || _sinceForecast >= reforecastEvery);

    return PipelineResult(
      globalIndex: idx,
      score: score,
      flagged: flagged,
      needForecast: needForecast,
    );
  }

  /// Installs a fresh model output. The window currently ends at
  /// `_globalIndex - 1`, so horizon step 0 predicts `_globalIndex` — the very
  /// next sample to arrive.
  void applyForecast(Float32List raw) {
    forecast = decodeForecast(raw, anchorIndex: _globalIndex);
    _sinceForecast = 0;
  }
}
