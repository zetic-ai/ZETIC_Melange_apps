import 'package:flutter_test/flutter_test.dart';
import 'package:siteguard/models/detection.dart';
import 'package:siteguard/services/postprocessor.dart';

import 'test_helpers.dart';

void main() {
  group('per-class threshold boundaries (Hardhat 0.25, others 0.15)', () {
    test('threshold map holds the GATE-2-approved operating points', () {
      expect(kClassThresholds[kClassHardhat], 0.25);
      expect(kClassThresholds[kClassVest], 0.15);
      expect(kClassThresholds[kClassNoHardhat], 0.15);
      expect(kClassThresholds[kClassNoVest], 0.15);
      expect(kClassThresholds, hasLength(4));
    });

    for (final (classId, name, threshold) in [
      (kClassHardhat, 'Hardhat', 0.25),
      (kClassVest, 'Safety Vest', 0.15),
      (kClassNoHardhat, 'NO-Hardhat', 0.15),
      (kClassNoVest, 'NO-Safety Vest', 0.15),
    ]) {
      test('$name: just-below ($threshold - 0.001) is dropped', () {
        final out = emptyOutput();
        setAnchor(out, 30,
            cx: 320, cy: 320, w: 100, h: 100,
            scores: {classId: threshold - 0.001});
        expect(postprocessOutput(identityRequest(out)), isEmpty);
      });

      test('$name: just-above ($threshold + 0.001) is kept', () {
        final out = emptyOutput();
        setAnchor(out, 31,
            cx: 320, cy: 320, w: 100, h: 100,
            scores: {classId: threshold + 0.001});
        final dets = postprocessOutput(identityRequest(out));
        expect(dets, hasLength(1));
        expect(dets.single.classId, classId);
      });
    }

    test(
        'per-class, not global: 0.20 passes for Vest but the same 0.20 '
        'fails for Hardhat', () {
      final outVest = emptyOutput();
      setAnchor(outVest, 32,
          cx: 200, cy: 200, w: 80, h: 80, scores: {kClassVest: 0.20});
      expect(postprocessOutput(identityRequest(outVest)), hasLength(1));

      final outHat = emptyOutput();
      setAnchor(outHat, 33,
          cx: 200, cy: 200, w: 80, h: 80, scores: {kClassHardhat: 0.20});
      expect(postprocessOutput(identityRequest(outHat)), isEmpty);
    });
  });
}
