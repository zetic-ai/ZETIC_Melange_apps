import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sensorforecastts/models/forecast.dart';
import 'package:sensorforecastts/services/postprocessor.dart';

Forecast _forecastWithBand(int t, double q10, double q90) {
  final raw = Float32List(kNumQuantiles * kHorizon);
  raw[0 * kHorizon + t] = q10;
  raw[8 * kHorizon + t] = q90;
  return decodeForecast(raw, anchorIndex: 0);
}

void main() {
  group('anomalyScore (band exceedance / IQR)', () {
    test('inside the band -> exactly 0 (clamped)', () {
      final f = _forecastWithBand(3, 10.0, 20.0);
      expect(anomalyScore(f, 3, 15.0), 0.0);
      expect(anomalyScore(f, 3, 10.0), 0.0, reason: 'on lower edge');
      expect(anomalyScore(f, 3, 20.0), 0.0, reason: 'on upper edge');
    });

    test('above band: (x - q90) / iqr', () {
      final f = _forecastWithBand(0, 10.0, 20.0); // iqr = 10
      expect(anomalyScore(f, 0, 25.0), closeTo(0.5, 1e-9));
      expect(anomalyScore(f, 0, 30.0), closeTo(1.0, 1e-9));
      expect(anomalyScore(f, 0, 40.0), closeTo(2.0, 1e-9));
    });

    test('below band: (q10 - x) / iqr', () {
      final f = _forecastWithBand(7, 10.0, 20.0);
      expect(anomalyScore(f, 7, 5.0), closeTo(0.5, 1e-9));
      expect(anomalyScore(f, 7, 0.0), closeTo(1.0, 1e-9));
    });

    test('degenerate zero-width band: iqr floor, no NaN/Inf', () {
      final f = _forecastWithBand(1, 42.0, 42.0); // width 0 -> floor 1e-6
      final s = anomalyScore(f, 1, 43.0);
      expect(s.isFinite, isTrue);
      expect(s, closeTo(1.0 / kIqrFloor, 1e6),
          reason: '1.0 excess over a floored band');
      expect(anomalyScore(f, 1, 42.0), 0.0);
    });

    test('inverted band (q90 < q10, pathological) still finite', () {
      final f = _forecastWithBand(2, 20.0, 10.0);
      expect(anomalyScore(f, 2, 15.0).isFinite, isTrue);
    });
  });
}
