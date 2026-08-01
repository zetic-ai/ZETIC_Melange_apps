import 'package:flutter_test/flutter_test.dart';
import 'package:vehicleplateyolo/models/detection.dart';
import 'package:vehicleplateyolo/services/nms.dart';

/// TRAP: suppression semantics. This model has ONE class, so GLOBAL NMS is
/// correct: overlapping plate boxes must collapse to the highest-confidence
/// one, while a distant plate must survive.
void main() {
  Detection box(double l, double t, double r, double b, double c) =>
      Detection(left: l, top: t, right: r, bottom: b, confidence: c);

  test('two overlapping boxes collapse to one (global NMS)', () {
    final a = box(0, 0, 100, 100, 0.9);
    final b = box(10, 10, 105, 105, 0.8); // IoU ~0.74 with a
    final kept = nonMaxSuppression([a, b], iouThreshold: 0.45);
    expect(kept.length, 1);
    expect(kept.single.confidence, closeTo(0.9, 1e-9)); // higher-conf survives
  });

  test('distant non-overlapping box survives', () {
    final a = box(0, 0, 50, 50, 0.9);
    final c = box(400, 400, 460, 460, 0.8); // IoU 0 with a
    final kept = nonMaxSuppression([a, c], iouThreshold: 0.45);
    expect(kept.length, 2);
  });
}
