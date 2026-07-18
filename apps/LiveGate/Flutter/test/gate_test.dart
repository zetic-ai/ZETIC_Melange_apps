import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:livegate/models/gate_result.dart';
import 'package:livegate/services/gate.dart';

/// Logits whose softmax[class 1] equals [p] exactly (classes 0 and 2 set to 0,
/// class 1 set to log(2p/(1-p))): softmax[1] = e^t / (2 + e^t) = p.
List<double> logitsForLive(double p) {
  final t = math.log(2 * p / (1 - p));
  return [0.0, t, 0.0];
}

/// A unit vector whose cosine with [1,0,0,...] equals [target].
Float32List probeWithCosine(double target, int dim) {
  final v = Float32List(dim);
  v[0] = target;
  if (dim > 1) v[1] = math.sqrt(1 - target * target);
  return v; // already unit length
}

Float32List unitFirst(int dim) {
  final v = Float32List(dim);
  v[0] = 1.0;
  return v;
}

void main() {
  group('T2 PAD live-class index (LIVE = softmax[1], not [0])', () {
    test('class 1 dominant => live; class 0 dominant => spoof', () {
      expect(livenessScore([0.0, 6.0, 0.0]), greaterThan(0.9));
      expect(livenessScore([6.0, 0.0, 0.0]), lessThan(0.1));
      // Decoder must read index 1, so a class-0 spike is NOT liveness.
      final v = composeVerdict(padLogits: [6.0, 0.0, 0.0]);
      expect(v.live, isFalse);
      expect(v.decision, GateDecision.spoof);
    });
  });

  group('T5 FACE L2-normalization semantics', () {
    test('unit(v)·unit(v) == 1, cosine in [-1,1], raw dot != cosine', () {
      final v = Float32List.fromList([3, 0, 4, 0]); // norm 5, not unit
      final u = l2normalize(v);
      expect(cosine(u, u), closeTo(1.0, 1e-6));

      final a = l2normalize(Float32List.fromList([1, 2, 3, 4]));
      final b = l2normalize(Float32List.fromList([4, 3, 2, 1]));
      final c = cosine(a, b);
      expect(c, inInclusiveRange(-1.0, 1.0));

      // Un-normalized dot differs from the cosine of the normalized vectors.
      var rawDot = 0.0;
      final ra = [1.0, 2, 3, 4], rb = [4.0, 3, 2, 1];
      for (var i = 0; i < 4; i++) {
        rawDot += ra[i] * rb[i];
      }
      expect(rawDot, isNot(closeTo(c, 1e-3)));
    });
  });

  group('T7 threshold boundaries', () {
    test('PAD liveness 0.45 boundary', () {
      expect(composeVerdict(padLogits: logitsForLive(0.4499)).live, isFalse);
      expect(composeVerdict(padLogits: logitsForLive(0.4501)).live, isTrue);
    });

    test('FACE cosine 0.363 boundary', () {
      final live = logitsForLive(0.9);
      final enrolled = unitFirst(128);
      final below = composeVerdict(
        padLogits: live,
        probeEmbedding: probeWithCosine(0.3629, 128),
        enrolledNormalized: enrolled,
      );
      final above = composeVerdict(
        padLogits: live,
        probeEmbedding: probeWithCosine(0.3631, 128),
        enrolledNormalized: enrolled,
      );
      expect(below.decision, GateDecision.liveNoMatch);
      expect(above.decision, GateDecision.pass);
    });
  });

  group('T8 gate composition (SPOOF never passes)', () {
    final enrolled = unitFirst(128);
    final matchingProbe = probeWithCosine(0.9, 128);

    test('SPOOF + high cosine => reject, score withheld', () {
      final v = composeVerdict(
        padLogits: logitsForLive(0.10), // spoof
        probeEmbedding: matchingProbe,
        enrolledNormalized: enrolled,
      );
      expect(v.decision, GateDecision.spoof);
      expect(v.passed, isFalse);
      expect(v.cosine, isNull, reason: 'match score hidden for a spoof');
    });

    test('LIVE + high cosine => pass', () {
      final v = composeVerdict(
        padLogits: logitsForLive(0.9),
        probeEmbedding: matchingProbe,
        enrolledNormalized: enrolled,
      );
      expect(v.decision, GateDecision.pass);
      expect(v.passed, isTrue);
    });

    test('LIVE + low cosine => live no-match', () {
      final v = composeVerdict(
        padLogits: logitsForLive(0.9),
        probeEmbedding: probeWithCosine(0.1, 128),
        enrolledNormalized: enrolled,
      );
      expect(v.decision, GateDecision.liveNoMatch);
    });

    test('LIVE + not enrolled => prompt enrollment', () {
      final v = composeVerdict(padLogits: logitsForLive(0.9));
      expect(v.decision, GateDecision.liveNoReference);
    });
  });
}
