import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:siteguard/models/detection.dart';
import 'package:siteguard/services/postprocessor.dart';

import 'test_helpers.dart';

double sigmoid(double x) => 1 / (1 + math.exp(-x));

void main() {
  group('score semantics: sigmoid is ALREADY applied in the ONNX head', () {
    test('confidence passes through exactly (no second sigmoid)', () {
      final out = emptyOutput();
      setAnchor(out, 50,
          cx: 320, cy: 320, w: 100, h: 100, scores: {kClassHardhat: 0.9});

      final dets = postprocessOutput(identityRequest(out));

      expect(dets, hasLength(1));
      // Must be the raw score (1e-6 tolerance: 0.9 is not exactly
      // representable in the float32 output buffer). A double-sigmoid bug
      // would emit sigmoid(0.9) = 0.711 instead.
      expect(dets.single.confidence, closeTo(0.9, 1e-6));
      expect(dets.single.confidence, isNot(closeTo(sigmoid(0.9), 1e-3)));
    });

    test('a raw score of 0.2 for Hardhat stays below its 0.25 threshold', () {
      // If a spurious sigmoid were applied, 0.2 -> 0.55 and this anchor would
      // wrongly pass the 0.25 threshold.
      final out = emptyOutput();
      setAnchor(out, 51,
          cx: 320, cy: 320, w: 100, h: 100, scores: {kClassHardhat: 0.2});

      expect(postprocessOutput(identityRequest(out)), isEmpty);
    });

    test('scores are already in 0..1 (documented invariant)', () {
      // Guards the documented contract: Stage-0 verified the exported ONNX
      // emits sigmoid-space scores (range 0..0.12 on random input). The
      // decoder must treat values as probabilities, so a score of exactly 1.0
      // must emit confidence 1.0, not sigmoid(1.0)=0.73.
      final out = emptyOutput();
      setAnchor(out, 52,
          cx: 320, cy: 320, w: 100, h: 100, scores: {kClassVest: 1.0});

      final dets = postprocessOutput(identityRequest(out));
      expect(dets.single.confidence, 1.0);
    });
  });
}
