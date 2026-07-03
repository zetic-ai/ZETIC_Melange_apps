import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sensorforecastts/models/forecast.dart';
import 'package:sensorforecastts/services/pipeline.dart';
import 'package:sensorforecastts/services/preprocessor.dart';

/// Raw tensor where band at horizon step t is [t, t+10] (q10 row = t,
/// q90 row = t+10), so the expected step is recoverable from the score.
Float32List _steppedBandRaw() {
  final raw = Float32List(kNumQuantiles * kHorizon);
  for (var t = 0; t < kHorizon; t++) {
    raw[0 * kHorizon + t] = t.toDouble(); // q10
    raw[8 * kHorizon + t] = t + 10.0; // q90 (iqr = 10)
  }
  return raw;
}

void main() {
  group('forecast/time alignment', () {
    test('horizon step 0 predicts the NEXT sample after the window', () {
      final p = ForecastPipeline();
      for (var i = 0; i < kContextLength; i++) {
        p.push(0.0);
      }
      p.applyForecast(_steppedBandRaw());
      expect(p.forecast!.anchorIndex, kContextLength,
          reason: 'window ended at 511, so step 0 predicts index 512');
      // Sample at global index 512 is horizon step 0: band [0,10], iqr 10.
      // x = 110 -> score (110-10)/10 = 10.
      final r = p.push(110.0);
      expect(r.globalIndex, kContextLength);
      expect(r.score, isNotNull);
      expect(r.score!, closeTo(10.0, 1e-6),
          reason: 'must score against step 0, not step 1');
    });

    test('subsequent samples map to steps 1,2,... of the same forecast', () {
      final p = ForecastPipeline(reforecastEvery: 1000000);
      for (var i = 0; i < kContextLength; i++) {
        p.push(0.0);
      }
      p.applyForecast(_steppedBandRaw());
      // Step t has band [t, t+10]. Feed x exactly on each upper edge: score 0.
      for (var t = 0; t < 5; t++) {
        final r = p.push(t + 10.0);
        expect(r.score, closeTo(0.0, 1e-6),
            reason: 'sample ${kContextLength + t} sits on step-$t upper edge');
      }
      // Now overshoot step 5's q90 (15) by exactly one band width: score 1.
      final r5 = p.push(25.0);
      expect(r5.score, closeTo(1.0, 1e-6));
    });

    test('re-forecast requested every 8 samples and fan re-anchors', () {
      final p = ForecastPipeline(); // reforecastEvery: 8 default
      PipelineResult? last;
      for (var i = 0; i < kContextLength; i++) {
        last = p.push(0.0);
      }
      // Window just filled with no forecast yet -> ask immediately.
      expect(last!.needForecast, isTrue);
      p.applyForecast(_steppedBandRaw());
      final anchor1 = p.forecast!.anchorIndex;

      // Next 7 pushes: no re-forecast due; 8th asks again.
      for (var k = 1; k <= 7; k++) {
        expect(p.push(5.0).needForecast, isFalse, reason: 'push $k of 8');
      }
      expect(p.push(5.0).needForecast, isTrue, reason: '8th push re-forecasts');
      p.applyForecast(_steppedBandRaw());
      final anchor2 = p.forecast!.anchorIndex;
      expect(anchor2, anchor1 + 8,
          reason: 'new fan anchors 8 samples later than the old one');

      // Sample after the NEW forecast scores against ITS step 0 (band [0,10]).
      final r = p.push(10.0);
      expect(r.score, closeTo(0.0, 1e-6),
          reason: 'scored against re-anchored step 0, not the old step 8');
    });

    test('no scoring before any forecast exists', () {
      final p = ForecastPipeline();
      final r = p.push(123.0);
      expect(r.score, isNull);
      expect(r.flagged, isFalse);
    });

    test('samples beyond the 64-step horizon are not scored', () {
      final p = ForecastPipeline(reforecastEvery: 1000000);
      for (var i = 0; i < kContextLength; i++) {
        p.push(0.0);
      }
      p.applyForecast(_steppedBandRaw());
      for (var t = 0; t < kHorizon; t++) {
        expect(p.push(0.0).score, isNotNull, reason: 'step $t in horizon');
      }
      expect(p.push(0.0).score, isNull,
          reason: 'step 64 is out of range -> stale forecast never scores');
    });
  });
}
