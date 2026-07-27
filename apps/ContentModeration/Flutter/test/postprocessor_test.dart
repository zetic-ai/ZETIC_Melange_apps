import 'dart:math' as math;

import 'package:contentmoderation/models/moderation_result.dart';
import 'package:contentmoderation/services/postprocessor.dart';
import 'package:flutter_test/flutter_test.dart';

/// Tier A — score semantics, label order, and decision-band correctness.
/// These traps do not throw when wrong; they produce plausible-but-inverted or
/// mis-banded output. See SPEC_STUB.md "Validation focus".
void main() {
  const post = Postprocessor(); // defaults: review 0.30, block 0.70

  group('softmax correctness (raw logits -> probability, applied exactly once)', () {
    test('P0 + P1 == 1 and matches the analytic sigmoid', () {
      final probs = Postprocessor.softmax([2.0, -1.0]);
      expect(probs[0] + probs[1], closeTo(1.0, 1e-12));
      // For 2 classes, softmax([a,b])[0] == sigmoid(a-b).
      final expectedP0 = 1.0 / (1.0 + math.exp(-(2.0 - (-1.0))));
      expect(probs[0], closeTo(expectedP0, 1e-12));
    });

    test('softmax is NOT double-applied (would compress toward 0.5)', () {
      final once = Postprocessor.softmax([4.0, -4.0]);
      final twice = Postprocessor.softmax(once);
      expect(twice[0], isNot(closeTo(once[0], 1e-6)));
      expect(post.classify([4.0, -4.0]).pNsfw, closeTo(once[0], 1e-12));
    });

    test('a plain sigmoid on the NSFW logit is NOT what we do', () {
      // sigmoid(logit0) ignores logit1; softmax uses both. On [1.0, 3.0]:
      final sigmoidOnly = 1.0 / (1.0 + math.exp(-1.0)); // 0.731...
      final p = post.classify([1.0, 3.0]).pNsfw; // softmax -> ~0.119
      expect(p, isNot(closeTo(sigmoidOnly, 0.1)));
      expect(p, closeTo(Postprocessor.softmax([1.0, 3.0])[0], 1e-12));
    });

    test('numerically stable for huge logits (no NaN/overflow)', () {
      final probs = Postprocessor.softmax([1000.0, -1000.0]);
      expect(probs[0], closeTo(1.0, 1e-9));
      expect(probs[1], closeTo(0.0, 1e-9));
      expect(probs[0].isNaN, isFalse);
    });

    test('rejects a wrong-length logit vector', () {
      expect(() => Postprocessor.softmax([1.0]), throwsArgumentError);
      expect(() => Postprocessor.softmax([1.0, 2.0, 3.0]), throwsArgumentError);
    });
  });

  group('label order — index 0 = NSFW, index 1 = SFW (REVERSED convention)', () {
    test('bigger logit at index 0 -> high P(NSFW) -> BLOCK', () {
      final r = post.classify([5.0, 0.0]);
      expect(r.pNsfw, greaterThan(0.95));
      expect(r.pSfw, lessThan(0.05));
      expect(r.decision, Decision.block);
    });

    test('bigger logit at index 1 -> low P(NSFW) -> KEEP', () {
      final r = post.classify([0.0, 5.0]);
      expect(r.pNsfw, lessThan(0.05));
      expect(r.pSfw, greaterThan(0.95));
      expect(r.decision, Decision.keep);
    });

    test('P(SFW) == 1 - P(NSFW)', () {
      final r = post.classify([1.3, -0.4]);
      expect(r.pSfw, closeTo(1.0 - r.pNsfw, 1e-12));
    });
  });

  group('decision bands at 0.30 / 0.70 (inclusive-lower)', () {
    // Build logits that yield an exact target P(NSFW): logit0 = ln(p/(1-p)),
    // logit1 = 0 -> softmax[0] = p.
    List<double> logitsFor(double p) => [math.log(p / (1 - p)), 0.0];

    test('just below 0.30 is KEEP', () {
      expect(post.classify(logitsFor(0.2999)).decision, Decision.keep);
    });
    test('exactly 0.30 is REVIEW (>= lower bound)', () {
      final r = post.classify(logitsFor(0.30));
      expect(r.pNsfw, closeTo(0.30, 1e-9));
      expect(r.decision, Decision.review);
    });
    test('just above 0.30 is REVIEW', () {
      expect(post.classify(logitsFor(0.3001)).decision, Decision.review);
    });
    test('just below 0.70 is REVIEW', () {
      expect(post.classify(logitsFor(0.6999)).decision, Decision.review);
    });
    test('exactly 0.70 is BLOCK (>= upper bound)', () {
      final r = post.classify(logitsFor(0.70));
      expect(r.pNsfw, closeTo(0.70, 1e-9));
      expect(r.decision, Decision.block);
    });
    test('just above 0.70 is BLOCK', () {
      expect(post.classify(logitsFor(0.7001)).decision, Decision.block);
    });

    test('bandFor is monotonic across the full range', () {
      expect(post.bandFor(0.0), Decision.keep);
      expect(post.bandFor(0.5), Decision.review);
      expect(post.bandFor(1.0), Decision.block);
    });
  });

  group('blur semantics (REVIEW/BLOCK blur, KEEP does not)', () {
    test('KEEP does not blur; REVIEW and BLOCK do', () {
      expect(Decision.keep.shouldBlur, isFalse);
      expect(Decision.review.shouldBlur, isTrue);
      expect(Decision.block.shouldBlur, isTrue);
    });
  });
}
