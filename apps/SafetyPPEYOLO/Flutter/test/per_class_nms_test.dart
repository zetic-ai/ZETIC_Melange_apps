import 'dart:ui' show Rect;

import 'package:flutter_test/flutter_test.dart';
import 'package:siteguard/models/detection.dart';
import 'package:siteguard/services/nms.dart';
import 'package:siteguard/services/postprocessor.dart';

import 'test_helpers.dart';

void main() {
  group('per-class (not global) NMS', () {
    test(
        'overlapping Hardhat + Safety Vest boxes BOTH survive '
        '(anti-global-NMS assertion)', () {
      // Two heavily overlapping boxes of DIFFERENT classes (IoU ~0.83).
      final a = Detection(
        rect: const Rect.fromLTRB(0.30, 0.30, 0.60, 0.60),
        classId: kClassHardhat,
        confidence: 0.9,
      );
      final b = Detection(
        rect: const Rect.fromLTRB(0.32, 0.30, 0.60, 0.60),
        classId: kClassVest,
        confidence: 0.5,
      );
      expect(iou(a.rect, b.rect), greaterThan(kIouThreshold),
          reason: 'test precondition: boxes must overlap above NMS IoU');

      // Per-class NMS: both survive.
      final kept = nmsPerClass([a, b], kIouThreshold);
      expect(kept, hasLength(2),
          reason: 'different-class overlaps must never suppress each other');

      // The SAME input under a hypothetical GLOBAL NMS (single-bucket call)
      // would kill the weaker box — proving the two behaviors differ and we
      // ship the per-class one.
      final globalKept = nonMaxSuppression([a, b], kIouThreshold);
      expect(globalKept, hasLength(1));
      expect(globalKept.single.classId, kClassHardhat);
    });

    test('two overlapping SAME-class boxes: only the stronger survives', () {
      final strong = Detection(
        rect: const Rect.fromLTRB(0.30, 0.30, 0.60, 0.60),
        classId: kClassHardhat,
        confidence: 0.9,
      );
      final weak = Detection(
        rect: const Rect.fromLTRB(0.31, 0.30, 0.60, 0.60),
        classId: kClassHardhat,
        confidence: 0.6,
      );
      final kept = nmsPerClass([strong, weak], kIouThreshold);
      expect(kept, hasLength(1));
      expect(kept.single.confidence, 0.9);
    });

    test('same-class boxes below the IoU threshold both survive', () {
      final a = Detection(
        rect: const Rect.fromLTRB(0.10, 0.10, 0.30, 0.30),
        classId: kClassVest,
        confidence: 0.9,
      );
      final b = Detection(
        rect: const Rect.fromLTRB(0.50, 0.50, 0.80, 0.80),
        classId: kClassVest,
        confidence: 0.8,
      );
      expect(nmsPerClass([a, b], kIouThreshold), hasLength(2));
    });

    test('end-to-end through postprocessOutput: hardhat+vest overlap kept',
        () {
      final out = emptyOutput();
      // Same worker: hardhat box and vest box almost coincident.
      setAnchor(out, 100,
          cx: 320, cy: 320, w: 200, h: 300, scores: {kClassHardhat: 0.85});
      setAnchor(out, 200,
          cx: 322, cy: 322, w: 200, h: 300, scores: {kClassVest: 0.60});

      final dets = postprocessOutput(identityRequest(out));
      expect(dets, hasLength(2));
      expect(
          dets.map((d) => d.classId).toSet(), {kClassHardhat, kClassVest});
    });
  });
}
