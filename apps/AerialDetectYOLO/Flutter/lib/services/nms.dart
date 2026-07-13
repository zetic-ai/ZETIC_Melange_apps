import '../models/detection.dart';

/// Intersection-over-union of two boxes (source-pixel space).
double iou(Detection a, Detection b) {
  final double ix1 = a.left > b.left ? a.left : b.left;
  final double iy1 = a.top > b.top ? a.top : b.top;
  final double ix2 = a.right < b.right ? a.right : b.right;
  final double iy2 = a.bottom < b.bottom ? a.bottom : b.bottom;

  final double iw = ix2 - ix1;
  final double ih = iy2 - iy1;
  if (iw <= 0 || ih <= 0) return 0.0;

  final double inter = iw * ih;
  final double union = a.area + b.area - inter;
  if (union <= 0) return 0.0;
  return inter / union;
}

/// Per-class greedy NMS.
///
/// Suppression is bucketed by [Detection.classId], so two overlapping boxes of
/// DIFFERENT classes both survive — only same-class overlaps are suppressed.
/// This is deliberately not global NMS (a person standing by a car must not
/// suppress the car). Sort once, precompute is implicit via [Detection.area].
List<Detection> nonMaxSuppressionPerClass(
  List<Detection> detections,
  double iouThreshold,
) {
  if (detections.length <= 1) return List<Detection>.of(detections);

  // Bucket by class.
  final Map<int, List<Detection>> byClass = <int, List<Detection>>{};
  for (final Detection d in detections) {
    (byClass[d.classId] ??= <Detection>[]).add(d);
  }

  final List<Detection> kept = <Detection>[];
  for (final List<Detection> bucket in byClass.values) {
    // Sort by confidence descending (once per bucket).
    bucket.sort((Detection a, Detection b) => b.confidence.compareTo(a.confidence));
    final List<bool> suppressed = List<bool>.filled(bucket.length, false);
    for (int i = 0; i < bucket.length; i++) {
      if (suppressed[i]) continue;
      final Detection a = bucket[i];
      kept.add(a);
      for (int j = i + 1; j < bucket.length; j++) {
        if (suppressed[j]) continue;
        if (iou(a, bucket[j]) > iouThreshold) {
          suppressed[j] = true;
        }
      }
    }
  }
  return kept;
}
