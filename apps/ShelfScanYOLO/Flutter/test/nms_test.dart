import 'package:flutter_test/flutter_test.dart';
import 'package:shelfscanyolo/models/detection.dart';
import 'package:shelfscanyolo/services/nms.dart';

void main() {
  group('single-class global NMS, IoU 0.45', () {
    test('strongly overlapping boxes -> lower confidence suppressed', () {
      final a = Detection(const BBox(0, 0, 100, 100), 0.90);
      final b = Detection(const BBox(10, 10, 100, 100), 0.80); // IoU 0.81
      final kept = nonMaxSuppression([b, a], 0.45);
      expect(kept, hasLength(1));
      expect(kept.single.confidence, 0.90); // the higher-conf box wins
    });

    test('weakly overlapping boxes -> both survive', () {
      final a = Detection(const BBox(0, 0, 100, 100), 0.90);
      final c = Detection(const BBox(70, 70, 170, 170), 0.80); // IoU ~0.047
      final kept = nonMaxSuppression([a, c], 0.45);
      expect(kept, hasLength(2));
    });

    test('global (not per-class): overlapping boxes always collapse to one', () {
      // There is a single class, so two overlapping facings are one product.
      final a = Detection(const BBox(0, 0, 50, 50), 0.7);
      final b = Detection(const BBox(2, 2, 50, 50), 0.6);
      expect(nonMaxSuppression([a, b], 0.45), hasLength(1));
    });

    test('suppression boundary is strict > threshold', () {
      final a = Detection(const BBox(0, 0, 100, 100), 0.9);
      // Choose an overlap and read its exact IoU.
      final b = Detection(const BBox(50, 0, 150, 100), 0.8);
      final iou = a.box.iou(b.box); // exact value both boxes share
      // threshold == iou -> NOT strictly greater -> both survive
      expect(nonMaxSuppression([a, b], iou), hasLength(2));
      // threshold just below iou -> suppressed
      expect(nonMaxSuppression([a, b], iou - 1e-6), hasLength(1));
    });

    test('empty and single inputs are safe', () {
      expect(nonMaxSuppression(const [], 0.45), isEmpty);
      final one = [Detection(const BBox(0, 0, 1, 1), 0.5)];
      expect(nonMaxSuppression(one, 0.45), hasLength(1));
    });

    test('keeps a dense cluster of non-overlapping facings', () {
      // 5x5 grid of tight, non-touching boxes -> all 25 survive.
      final dets = <Detection>[];
      for (var r = 0; r < 5; r++) {
        for (var c = 0; c < 5; c++) {
          dets.add(Detection(
            BBox(c * 20.0, r * 20.0, c * 20.0 + 18, r * 20.0 + 18),
            0.5 + 0.01 * (r * 5 + c),
          ));
        }
      }
      expect(nonMaxSuppression(dets, 0.45), hasLength(25));
    });
  });
}
