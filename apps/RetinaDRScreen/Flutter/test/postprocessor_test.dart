import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:retinadrscreen/services/postprocessor.dart';

void main() {
  const post = Postprocessor(); // default threshold 0.5

  group('softmax correctness (2-logit -> probability)', () {
    test('P0 + P1 == 1 and matches the analytic sigmoid', () {
      final probs = Postprocessor.softmax([2.0, -1.0]);
      expect(probs[0] + probs[1], closeTo(1.0, 1e-12));
      // For 2 classes, softmax([a,b])[1] == sigmoid(b-a).
      final expectedP1 = 1.0 / (1.0 + math.exp(2.0 - (-1.0)));
      expect(probs[1], closeTo(expectedP1, 1e-12));
    });

    test('verdict = argmax matches the larger logit', () {
      expect(post.classify([3.0, -3.0]).referable, isFalse); // logit0 bigger
      expect(post.classify([-3.0, 3.0]).referable, isTrue); // logit1 bigger
    });

    test('softmax is NOT double-applied', () {
      // Applying softmax twice would compress toward 0.5 and change the value.
      final once = Postprocessor.softmax([4.0, -4.0]);
      final twice = Postprocessor.softmax(once);
      expect(twice[1], isNot(closeTo(once[1], 1e-6)));
      // The classifier uses exactly one softmax: P(referable) must equal `once`.
      expect(post.classify([4.0, -4.0]).pReferable, closeTo(once[1], 1e-12));
    });

    test('numerically stable for large logits (no overflow to NaN)', () {
      final probs = Postprocessor.softmax([1000.0, -1000.0]);
      expect(probs[0], closeTo(1.0, 1e-9));
      expect(probs[1], closeTo(0.0, 1e-9));
      expect(probs[1].isNaN, isFalse);
    });

    test('rejects a wrong-length logit vector', () {
      expect(() => Postprocessor.softmax([1.0]), throwsArgumentError);
      expect(() => Postprocessor.softmax([1.0, 2.0, 3.0]), throwsArgumentError);
    });
  });

  group('threshold boundary at 0.5', () {
    test('exactly 0.5 (equal logits) is REFERABLE (>= threshold)', () {
      final r = post.classify([1.234, 1.234]);
      expect(r.pReferable, closeTo(0.5, 1e-12));
      expect(r.referable, isTrue);
    });

    test('just below 0.5 is NOT referable', () {
      // logit1 slightly smaller than logit0 -> P(referable) < 0.5.
      final r = post.classify([0.0001, 0.0]);
      expect(r.pReferable, lessThan(0.5));
      expect(r.referable, isFalse);
    });

    test('just above 0.5 is referable', () {
      final r = post.classify([0.0, 0.0001]);
      expect(r.pReferable, greaterThan(0.5));
      expect(r.referable, isTrue);
    });
  });

  group('label mapping (index 0 = Nrdr, index 1 = Rdr) — not inverted', () {
    test('bigger Nrdr logit -> NOT REFERABLE with low P(referable)', () {
      final r = post.classify([5.0, 0.0]);
      expect(r.referable, isFalse);
      expect(r.pReferable, lessThan(0.05));
      expect(r.verdictLabel, 'NOT REFERABLE');
    });

    test('bigger Rdr logit -> REFERABLE with high P(referable)', () {
      final r = post.classify([0.0, 5.0]);
      expect(r.referable, isTrue);
      expect(r.pReferable, greaterThan(0.95));
      expect(r.verdictLabel, 'REFERABLE');
    });

    test('confidence is max(P0, P1) of the shown verdict', () {
      final r = post.classify([5.0, 0.0]);
      expect(r.confidence, closeTo(r.pNotReferable, 1e-12));
      expect(r.confidence, greaterThanOrEqualTo(0.5));
    });
  });

  group('anti-degeneracy (healthy eye not over-flagged)', () {
    test('grade-0 demo logits [10.11, -0.66] -> NOT REFERABLE, P ~ 0', () {
      final r = post.classify([10.11, -0.66]);
      expect(r.referable, isFalse);
      expect(r.pReferable, closeTo(0.0, 1e-3));
    });
  });
}
