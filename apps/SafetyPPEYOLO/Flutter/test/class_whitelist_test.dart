import 'package:flutter_test/flutter_test.dart';
import 'package:siteguard/models/detection.dart';
import 'package:siteguard/services/postprocessor.dart';

import 'test_helpers.dart';

void main() {
  group('class whitelist {3,7,9,12} enforcement', () {
    test('whitelist constant is exactly the GATE-2 set', () {
      expect(kRenderedClassIds, {3, 7, 9, 12});
    });

    test('Person (11) is NEVER emitted, even at very high confidence', () {
      final out = emptyOutput();
      setAnchor(out, 60,
          cx: 320, cy: 320, w: 200, h: 400, scores: {kClassPerson: 0.99});

      expect(postprocessOutput(identityRequest(out)), isEmpty,
          reason: 'Person is a measured-degenerate class and must never '
              'be rendered (SPEC.md)');
    });

    for (final (classId, name) in [
      (0, 'Fall-Detected'),
      (1, 'Gloves'),
      (2, 'Goggles'),
      (4, 'Mask (excluded by GATE-2 ruling)'),
      (5, 'NO-Gloves'),
      (6, 'NO-Goggles'),
      (8, 'NO-Mask (excluded by GATE-2 ruling)'),
      (10, 'No_Harness'),
      (11, 'Person (degenerate)'),
    ]) {
      test('non-whitelisted class $classId ($name) emits nothing at 0.95', () {
        final out = emptyOutput();
        setAnchor(out, 61,
            cx: 320, cy: 320, w: 100, h: 100, scores: {classId: 0.95});
        expect(postprocessOutput(identityRequest(out)), isEmpty);
      });
    }

    test(
        'argmax semantics: when Person outscores Hardhat on the SAME anchor, '
        'the anchor is dropped (not relabeled as Hardhat)', () {
      final out = emptyOutput();
      setAnchor(out, 62,
          cx: 320, cy: 320, w: 100, h: 100,
          scores: {kClassPerson: 0.60, kClassHardhat: 0.40});

      expect(postprocessOutput(identityRequest(out)), isEmpty,
          reason: 'matches the Stage-0 harness: argmax over all 13 classes, '
              'then whitelist-filter the winner');
    });

    test(
        'but when Hardhat wins the argmax, the anchor is emitted even if a '
        'non-whitelisted class also scored', () {
      final out = emptyOutput();
      setAnchor(out, 63,
          cx: 320, cy: 320, w: 100, h: 100,
          scores: {kClassPerson: 0.30, kClassHardhat: 0.55});

      final dets = postprocessOutput(identityRequest(out));
      expect(dets, hasLength(1));
      expect(dets.single.classId, kClassHardhat);
    });
  });
}
