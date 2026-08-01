import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vehicleplateyolo/services/letterbox.dart';
import 'package:vehicleplateyolo/services/postprocessor.dart';

/// TRAP: Sigmoid is BAKED into the exported graph. Applying sigmoid again in
/// Dart silently degrades confidence (0.9 -> 0.711) and would push borderline
/// plates under the threshold. Confidence must pass through unchanged.
void main() {
  double sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

  test('confidence passed through unchanged (no re-applied sigmoid)', () {
    const n = 8400;
    final params = LetterboxParams.forImage(640, 640, 640);
    const post = Postprocessor();

    for (final conf in [0.90, 0.30]) {
      final out = Float32List(5 * n);
      const a = 10;
      out[0 * n + a] = 320;
      out[1 * n + a] = 320;
      out[2 * n + a] = 40;
      out[3 * n + a] = 40;
      out[4 * n + a] = conf;

      final d = post.decode(out, params).single;
      expect(d.confidence, closeTo(conf, 1e-6));
      // Prove it is NOT the double-sigmoid value.
      expect((d.confidence - sigmoid(conf)).abs(), greaterThan(0.05));
    }
  });
}
