import 'package:flutter_test/flutter_test.dart';
import 'package:sensorforecastts/services/postprocessor.dart';

void main() {
  group('AnomalyDetector (threshold + 2-consecutive debounce)', () {
    test('score just below threshold never flags', () {
      final d = AnomalyDetector();
      for (var i = 0; i < 10; i++) {
        expect(d.onScore(0.999), isFalse);
      }
    });

    test('single exceedance does not flag (debounce)', () {
      final d = AnomalyDetector();
      expect(d.onScore(1.001), isFalse, reason: 'first exceedance held back');
    });

    test('two consecutive exceedances flag on the second', () {
      final d = AnomalyDetector();
      expect(d.onScore(1.001), isFalse);
      expect(d.onScore(1.001), isTrue);
      expect(d.onScore(1.001), isTrue, reason: 'stays flagged while streak holds');
    });

    test('exceedance-gap-exceedance does NOT flag (streak resets)', () {
      final d = AnomalyDetector();
      expect(d.onScore(2.0), isFalse);
      expect(d.onScore(0.0), isFalse, reason: 'normal sample resets streak');
      expect(d.onScore(2.0), isFalse, reason: 'streak restarted at 1');
      expect(d.onScore(2.0), isTrue);
    });

    test('boundary semantics: exactly threshold counts (>=)', () {
      final d = AnomalyDetector();
      expect(d.onScore(1.0), isFalse);
      expect(d.onScore(1.0), isTrue);
    });

    test('threshold is adjustable (slider 0.5-3.0)', () {
      final d = AnomalyDetector(threshold: 3.0);
      d.onScore(2.9);
      d.onScore(2.9);
      expect(d.streak, 0, reason: '2.9 < 3.0 never streaks');
      d.threshold = 0.5;
      expect(d.onScore(0.6), isFalse);
      expect(d.onScore(0.6), isTrue);
    });

    test('reset() clears the streak', () {
      final d = AnomalyDetector();
      d.onScore(2.0);
      d.reset();
      expect(d.onScore(2.0), isFalse, reason: 'streak restarted after reset');
    });
  });
}
