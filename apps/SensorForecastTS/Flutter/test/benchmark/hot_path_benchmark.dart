import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sensorforecastts/models/forecast.dart';
import 'package:sensorforecastts/services/pipeline.dart';
import 'package:sensorforecastts/services/preprocessor.dart';

/// Tier A4 hot-path micro-benchmark (VALIDATION.md).
///
/// Measures the full pure-Dart per-TICK path (feed value -> window push ->
/// score -> debounce) and the per-FORECAST path (window snapshot -> decode ->
/// install), on mock tensors of the real shapes. This is the post-processing
/// budget only — NPU/SDK time is not measurable off-device.
///
/// Run:  flutter test test/benchmark/hot_path_benchmark.dart
void main() {
  test('hot-path micro-benchmark (median over batches)', () {
    final rng = math.Random(1);

    // Mock model output [1,9,64], quantile-major, plausible band.
    final raw = Float32List(kNumQuantiles * kHorizon);
    for (var t = 0; t < kHorizon; t++) {
      final mid = 88.0 + math.sin(t / 9.0) * 3.0;
      for (var q = 0; q < kNumQuantiles; q++) {
        raw[q * kHorizon + t] = mid + (q - 4) * 1.2;
      }
    }

    final pipeline = ForecastPipeline();
    for (var i = 0; i < kContextLength; i++) {
      pipeline.push(88.0 + rng.nextDouble());
    }
    pipeline.applyForecast(raw);

    double medianOf(List<double> xs) {
      final s = [...xs]..sort();
      return s[s.length ~/ 2];
    }

    // --- per-tick path: push + score + debounce ---------------------------
    const ticksPerBatch = 10000;
    const batches = 15;
    final tickNs = <double>[];
    for (var b = 0; b < batches; b++) {
      final sw = Stopwatch()..start();
      for (var i = 0; i < ticksPerBatch; i++) {
        final r = pipeline.push(88.0 + rng.nextDouble() * 2.0);
        if (r.needForecast) pipeline.applyForecast(raw);
      }
      sw.stop();
      tickNs.add(sw.elapsedMicroseconds * 1000.0 / ticksPerBatch);
    }

    // --- per-forecast path: snapshot + decode + install -------------------
    final window = pipeline.window;
    final dst = Float32List(kContextLength);
    const fcPerBatch = 2000;
    final fcNs = <double>[];
    for (var b = 0; b < batches; b++) {
      final sw = Stopwatch()..start();
      for (var i = 0; i < fcPerBatch; i++) {
        window.snapshotInto(dst);
        pipeline.applyForecast(raw);
      }
      sw.stop();
      fcNs.add(sw.elapsedMicroseconds * 1000.0 / fcPerBatch);
    }

    final tickMed = medianOf(tickNs);
    final fcMed = medianOf(fcNs);
    // Demo budget at 20 sps + re-forecast every 8 samples:
    final perSecondUs =
        (tickMed * 20 + fcMed * 2.5) / 1000.0; // ns -> us per second of demo
    // ignore: avoid_print
    print('HOT PATH per-tick median:     ${tickMed.toStringAsFixed(0)} ns');
    // ignore: avoid_print
    print('HOT PATH per-forecast median: ${fcMed.toStringAsFixed(0)} ns');
    // ignore: avoid_print
    print('HOT PATH dart budget/second:  ${perSecondUs.toStringAsFixed(1)} us');

    // Regression guards (generous: CI machines vary).
    expect(tickMed, lessThan(50 * 1000), reason: 'tick must stay < 50 us');
    expect(fcMed, lessThan(500 * 1000), reason: 'forecast prep < 500 us');
  });
}
