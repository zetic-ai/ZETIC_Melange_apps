import 'dart:math' as math;

import '../models/detection.dart';

/// Global (single-class) non-max suppression.
///
/// This model has ONE class (`license_plate`), so global NMS is correct: there
/// is no second class whose box must survive next to a plate (the per-class vs
/// global trap in VALIDATION.md A3). Sort once, precompute areas, then the
/// O(n^2) pass over an already-small survivor set (threshold ran first).
List<Detection> nonMaxSuppression(
  List<Detection> boxes, {
  double iouThreshold = 0.45,
}) {
  if (boxes.length <= 1) return List<Detection>.of(boxes);

  // Sort by confidence descending (once).
  final order = List<int>.generate(boxes.length, (i) => i)
    ..sort((a, b) => boxes[b].confidence.compareTo(boxes[a].confidence));

  // Precompute areas.
  final areas = List<double>.filled(boxes.length, 0);
  for (var i = 0; i < boxes.length; i++) {
    areas[i] = boxes[i].width * boxes[i].height;
  }

  final suppressed = List<bool>.filled(boxes.length, false);
  final kept = <Detection>[];

  for (var oi = 0; oi < order.length; oi++) {
    final i = order[oi];
    if (suppressed[i]) continue;
    kept.add(boxes[i]);
    final a = boxes[i];
    for (var oj = oi + 1; oj < order.length; oj++) {
      final j = order[oj];
      if (suppressed[j]) continue;
      if (_iou(a, boxes[j], areas[i], areas[j]) > iouThreshold) {
        suppressed[j] = true;
      }
    }
  }
  return kept;
}

double _iou(Detection a, Detection b, double areaA, double areaB) {
  final ix1 = math.max(a.left, b.left);
  final iy1 = math.max(a.top, b.top);
  final ix2 = math.min(a.right, b.right);
  final iy2 = math.min(a.bottom, b.bottom);
  final iw = ix2 - ix1;
  final ih = iy2 - iy1;
  if (iw <= 0 || ih <= 0) return 0.0;
  final inter = iw * ih;
  return inter / (areaA + areaB - inter);
}
