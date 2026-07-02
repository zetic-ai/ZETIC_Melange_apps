import 'package:dentalxraydetect/models/detection.dart';
import 'package:dentalxraydetect/services/nms.dart';
import 'package:flutter_test/flutter_test.dart';

Detection box(double l, double t, double r, double b, int cls, double conf) =>
    Detection(left: l, top: t, right: r, bottom: b, classId: cls, confidence: conf);

void main() {
  group('per-class NMS (IoU 0.45)', () {
    test('test_per_class_nms_keeps_overlapping_different_classes', () {
      // A caries box and an impacted-tooth box fully overlapping on the same
      // molar -> BOTH survive (per-class, not global).
      final List<Detection> dets = <Detection>[
        box(0, 0, 100, 100, 0, 0.9), // caries
        box(0, 0, 100, 100, 2, 0.8), // impacted_tooth
      ];
      final List<Detection> kept = nonMaxSuppressionPerClass(dets, 0.45);
      expect(kept.length, 2);
      expect(kept.map((Detection d) => d.classId).toSet(), <int>{0, 2});
    });

    test('test_nms_suppresses_same_class_overlap', () {
      final List<Detection> dets = <Detection>[
        box(0, 0, 100, 100, 0, 0.9),
        box(5, 5, 105, 105, 0, 0.8), // IoU ~0.82 with the first
      ];
      final List<Detection> kept = nonMaxSuppressionPerClass(dets, 0.45);
      expect(kept.length, 1);
      expect(kept.first.confidence, 0.9, reason: 'higher-confidence box kept');
    });

    test('test_nms_iou_boundary_at_0.45', () {
      // Horizontal overlap controls IoU. x=30 -> IoU 0.538 (> 0.45, suppress);
      // x=60 -> IoU 0.25 (<= 0.45, keep both).
      final List<Detection> suppress = nonMaxSuppressionPerClass(<Detection>[
        box(0, 0, 100, 100, 0, 0.9),
        box(30, 0, 130, 100, 0, 0.8),
      ], 0.45);
      expect(suppress.length, 1);

      final List<Detection> keep = nonMaxSuppressionPerClass(<Detection>[
        box(0, 0, 100, 100, 0, 0.9),
        box(60, 0, 160, 100, 0, 0.8),
      ], 0.45);
      expect(keep.length, 2);
    });

    test('iou helper is correct for a known overlap', () {
      // 50% horizontal overlap: inter=5000, union=15000 -> 1/3.
      expect(
        iou(box(0, 0, 100, 100, 0, 1), box(50, 0, 150, 100, 0, 1)),
        closeTo(1 / 3, 1e-9),
      );
      // Disjoint -> 0.
      expect(
        iou(box(0, 0, 10, 10, 0, 1), box(50, 50, 60, 60, 0, 1)),
        0.0,
      );
    });
  });
}
