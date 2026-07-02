import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:retinadrgrade/models/grading_result.dart';
import 'package:retinadrgrade/services/postprocessor.dart';

/// Build logits that softmax EXACTLY to a target distribution: logit_i = ln(p_i).
/// softmax(ln p) == p (since exp(ln p_i)=p_i and Σp_i=1). Lets us drive the
/// postprocessor from the DEMO_IMAGES.md measured 5-way softmax vectors.
List<double> logitsFor(List<double> probs) =>
    probs.map((p) => math.log(p)).toList();

void main() {
  const post = Postprocessor();

  group('softmax ONCE over 5 logits', () {
    test('Σ probs == 1 and length 5', () {
      final probs = Postprocessor.softmax([2.0, -1.0, 0.5, 3.0, -2.0]);
      expect(probs.length, 5);
      expect(probs.reduce((a, b) => a + b), closeTo(1.0, 1e-12));
    });

    test('matches the explicit exp/normalize definition', () {
      final logits = [1.0, 2.0, 3.0, 0.0, -1.0];
      final probs = Postprocessor.softmax(logits);
      final exps = logits.map(math.exp).toList();
      final sum = exps.reduce((a, b) => a + b);
      for (var i = 0; i < 5; i++) {
        expect(probs[i], closeTo(exps[i] / sum, 1e-12));
      }
    });

    test('is NOT double-applied (softmax of softmax differs)', () {
      final once = Postprocessor.softmax([6.0, -2.0, 0.0, 1.0, -4.0]);
      final twice = Postprocessor.softmax(once);
      // Double-applying compresses toward uniform (0.2 each) and changes values.
      expect(twice[0], isNot(closeTo(once[0], 1e-4)));
      // classify() applies exactly one softmax: probs must equal `once`.
      final r = post.classify([6.0, -2.0, 0.0, 1.0, -4.0]);
      for (var i = 0; i < 5; i++) {
        expect(r.perGradeProbs[i], closeTo(once[i], 1e-12));
      }
    });

    test('numerically stable for large logits (no overflow to NaN)', () {
      final probs = Postprocessor.softmax([1000.0, -1000.0, 0.0, 500.0, -500.0]);
      expect(probs.every((p) => p.isFinite), isTrue);
      expect(probs.reduce((a, b) => a + b), closeTo(1.0, 1e-9));
      expect(probs[0], closeTo(1.0, 1e-6)); // dominant logit wins
    });

    test('rejects a wrong-length logit vector', () {
      expect(() => Postprocessor.softmax([1.0, 2.0]), throwsArgumentError);
      expect(() => Postprocessor.softmax([1, 2, 3, 4, 5, 6].map((e) => e.toDouble()).toList()),
          throwsArgumentError);
    });
  });

  group('argmax -> grade with IDENTITY id2label (no remap)', () {
    test('the largest-logit index IS the grade, for every grade 0..4', () {
      for (var g = 0; g < 5; g++) {
        final logits = List<double>.filled(5, 0.0);
        logits[g] = 10.0; // make grade g dominate
        final r = post.classify(logits);
        expect(r.grade, g);
        expect(r.gradeLabel, GradingResult.gradeLabels[g]);
      }
    });

    test('a permutation of logits moves the grade correspondingly (not fixed)', () {
      expect(post.classify([9, 0, 0, 0, 0].map((e) => e.toDouble()).toList()).grade, 0);
      expect(post.classify([0, 0, 0, 0, 9].map((e) => e.toDouble()).toList()).grade, 4);
      expect(post.classify([0, 0, 9, 0, 0].map((e) => e.toDouble()).toList()).grade, 2);
    });

    test('argmax ties resolve to the lowest index (deterministic)', () {
      expect(Postprocessor.argmax([1.0, 1.0, 1.0, 1.0, 1.0]), 0);
    });
  });

  group('referable = grade >= 2 (exact boundary)', () {
    test('grade 0 (No DR) -> NOT referable', () {
      final r = post.classify([10, 0, 0, 0, 0].map((e) => e.toDouble()).toList());
      expect(r.grade, 0);
      expect(r.referable, isFalse);
    });

    test('grade 1 (Mild) -> NOT referable (boundary just below)', () {
      final r = post.classify([0, 10, 0, 0, 0].map((e) => e.toDouble()).toList());
      expect(r.grade, 1);
      expect(r.referable, isFalse);
    });

    test('grade 2 (Moderate) -> referable (boundary exactly at 2)', () {
      final r = post.classify([0, 0, 10, 0, 0].map((e) => e.toDouble()).toList());
      expect(r.grade, 2);
      expect(r.referable, isTrue);
    });

    test('grades 3 and 4 -> referable', () {
      expect(post.classify([0, 0, 0, 10, 0].map((e) => e.toDouble()).toList()).referable, isTrue);
      expect(post.classify([0, 0, 0, 0, 10].map((e) => e.toDouble()).toList()).referable, isTrue);
    });
  });

  group('anti-degeneracy / spread — reproduces DEMO_IMAGES.md distributions', () {
    test('g0 demo softmax -> grade 0, top ~0.982, NOT referable', () {
      final r = post.classify(logitsFor([0.982, 0.008, 0.005, 0.002, 0.003]));
      expect(r.grade, 0);
      expect(r.referable, isFalse);
      expect(r.topConfidence, closeTo(0.982, 5e-3));
      expect(r.perGradeProbs[0], closeTo(0.982, 5e-3));
    });

    test('g3 demo softmax -> grade 3, top ~0.810, referable', () {
      final r = post.classify(logitsFor([0.007, 0.006, 0.088, 0.810, 0.089]));
      expect(r.grade, 3);
      expect(r.referable, isTrue);
      expect(r.topConfidence, closeTo(0.810, 5e-3));
    });

    test('g4 demo softmax -> grade 4, top ~0.809, referable', () {
      final r = post.classify(logitsFor([0.019, 0.011, 0.035, 0.125, 0.809]));
      expect(r.grade, 4);
      expect(r.referable, isTrue);
      expect(r.topConfidence, closeTo(0.809, 5e-3));
    });

    test('distribution is not collapsed to one mode (spread across grades)', () {
      // The g3 case should place real mass on neighbouring grades (2 and 4),
      // not spike a single grade to ~1.0 — a collapse would signal a broken
      // resize/normalize/softmax pipeline.
      final r = post.classify(logitsFor([0.007, 0.006, 0.088, 0.810, 0.089]));
      expect(r.perGradeProbs[2], greaterThan(0.02));
      expect(r.perGradeProbs[4], greaterThan(0.02));
    });
  });

  group('GradingResult surface', () {
    test('gradeHeadline and referableLabel read correctly', () {
      final r = post.classify([0, 0, 0, 10, 0].map((e) => e.toDouble()).toList());
      expect(r.gradeHeadline, 'Grade 3 — Severe');
      expect(r.referableLabel, 'REFERABLE');
    });
  });
}
