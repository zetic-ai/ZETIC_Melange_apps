import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sensorforecastts/models/forecast.dart';
import 'package:sensorforecastts/services/postprocessor.dart';

void main() {
  group('quantile-major [1,9,64] decode', () {
    test('flat index is q*64+t, NOT t*9+q (sentinel probes)', () {
      final raw = Float32List(kNumQuantiles * kHorizon);
      // Unique sentinels at probed (q,t) positions, placed QUANTILE-MAJOR.
      const probes = <(int, int, double)>[
        (0, 0, 111.0),
        (7, 3, 222.0),
        (8, 63, 333.0),
        (4, 0, 444.0),
        (4, 63, 555.0),
        (2, 10, 666.0),
      ];
      for (final (q, t, v) in probes) {
        raw[q * kHorizon + t] = v;
      }
      final f = decodeForecast(raw, anchorIndex: 0);
      for (final (q, t, v) in probes) {
        expect(f.at(q, t), v, reason: '(q=$q,t=$t) must read flat q*64+t');
      }
      // Explicit cross-check: a row-major reader would find 222.0 at
      // t*9+q = 3*9+7 = 34 -> (q=0,t=34) in our layout. Assert it is NOT there.
      expect(f.at(0, 34), isNot(222.0),
          reason: 'row-major (t*9+q) reading must fail these probes');
    });

    test('median comes from row index 4', () {
      final raw = Float32List(kNumQuantiles * kHorizon);
      for (var t = 0; t < kHorizon; t++) {
        raw[kMedianRow * kHorizon + t] = 42.0 + t;
      }
      final f = decodeForecast(raw, anchorIndex: 0);
      expect(f.median(0), 42.0);
      expect(f.median(63), 42.0 + 63);
    });

    test('q10 is row 0 and q90 is row 8', () {
      final raw = Float32List(kNumQuantiles * kHorizon);
      raw[0 * kHorizon + 5] = -7.0; // q10 @ t=5
      raw[8 * kHorizon + 5] = 7.0; // q90 @ t=5
      final f = decodeForecast(raw, anchorIndex: 0);
      expect(f.q10(5), -7.0);
      expect(f.q90(5), 7.0);
    });

    test('decode copies the buffer (SDK view-reuse safety)', () {
      final raw = Float32List(kNumQuantiles * kHorizon);
      raw[0] = 1.0;
      final f = decodeForecast(raw, anchorIndex: 0);
      raw[0] = 99.0; // simulate the native buffer being overwritten
      expect(f.at(0, 0), 1.0, reason: 'Forecast must own a copy');
    });

    test('wrong-length tensor is rejected', () {
      expect(() => decodeForecast(Float32List(100), anchorIndex: 0),
          throwsA(isA<AssertionError>()));
    });
  });
}
