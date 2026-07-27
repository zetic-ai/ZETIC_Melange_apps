import 'dart:ui' show Rect;

import 'package:flutter_test/flutter_test.dart';
import 'package:snapseek/models/detection.dart';
import 'package:snapseek/services/nms.dart';
import 'package:snapseek/services/postprocessor.dart';

import 'test_helpers.dart';

void main() {
  group('global (class-agnostic) NMS', () {
    test('two overlapping boxes of DIFFERENT classes collapse to one', () {
      // A person box and a handbag box on the same figure (high IoU). A salient
      // localizer must treat them as ONE object → keep only the top-conf box.
      // (Per-class NMS would wrongly keep both.)
      final a = Detection(
          rect: const Rect.fromLTRB(0.10, 0.10, 0.50, 0.90),
          classId: kClassPerson,
          confidence: 0.9);
      final b = Detection(
          rect: const Rect.fromLTRB(0.12, 0.12, 0.52, 0.88),
          classId: kClassHandbag,
          confidence: 0.6);

      final kept = globalNms([a, b], kIouThreshold);
      expect(kept, hasLength(1));
      expect(kept.single.classId, kClassPerson);
      expect(kept.single.confidence, 0.9);
    });

    test('non-overlapping different-class boxes both survive', () {
      final a = Detection(
          rect: const Rect.fromLTRB(0.0, 0.0, 0.2, 0.2),
          classId: kClassPerson,
          confidence: 0.9);
      final b = Detection(
          rect: const Rect.fromLTRB(0.7, 0.7, 0.95, 0.95),
          classId: kClassBottle,
          confidence: 0.8);

      final kept = globalNms([a, b], kIouThreshold);
      expect(kept, hasLength(2));
    });

    test('survivors are sorted by descending confidence (primary first)', () {
      final lo = Detection(
          rect: const Rect.fromLTRB(0.0, 0.0, 0.1, 0.1),
          classId: 1,
          confidence: 0.3);
      final hi = Detection(
          rect: const Rect.fromLTRB(0.5, 0.5, 0.6, 0.6),
          classId: 2,
          confidence: 0.95);
      final kept = globalNms([lo, hi], kIouThreshold);
      expect(kept.first.confidence, 0.95);
    });

    test('end-to-end: two overlapping anchors of different classes → 1 box', () {
      final out = emptyOutput();
      setAnchor(out, 1,
          cx: 320, cy: 320, w: 200, h: 400, scores: {kClassPerson: 0.88});
      setAnchor(out, 2,
          cx: 322, cy: 320, w: 198, h: 396, scores: {kClassBackpack: 0.55});
      final dets = postprocessDetect(identityRequest(out));
      expect(dets, hasLength(1));
      expect(dets.single.classId, kClassPerson);
    });
  });
}
