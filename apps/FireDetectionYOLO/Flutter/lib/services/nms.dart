import 'dart:ui' show Rect;

import '../models/detection.dart';

/// Intersection-over-Union of two normalized rectangles.
double iou(Rect a, Rect b) {
  final double left = a.left > b.left ? a.left : b.left;
  final double top = a.top > b.top ? a.top : b.top;
  final double right = a.right < b.right ? a.right : b.right;
  final double bottom = a.bottom < b.bottom ? a.bottom : b.bottom;

  final double interW = right - left;
  final double interH = bottom - top;
  if (interW <= 0 || interH <= 0) return 0.0;

  final double interArea = interW * interH;
  final double union = a.width * a.height + b.width * b.height - interArea;
  if (union <= 0) return 0.0;
  return interArea / union;
}

/// Greedy non-maximum suppression over a single class.
///
/// Sorts by descending confidence, keeps the strongest box, and discards any
/// remaining box whose IoU with a kept box exceeds [iouThreshold].
List<Detection> nonMaxSuppression(
  List<Detection> detections,
  double iouThreshold,
) {
  if (detections.length <= 1) return detections;

  final sorted = List<Detection>.of(detections)
    ..sort((a, b) => b.confidence.compareTo(a.confidence));
  final removed = List<bool>.filled(sorted.length, false);
  final kept = <Detection>[];

  for (var i = 0; i < sorted.length; i++) {
    if (removed[i]) continue;
    kept.add(sorted[i]);
    for (var j = i + 1; j < sorted.length; j++) {
      if (removed[j]) continue;
      if (iou(sorted[i].rect, sorted[j].rect) > iouThreshold) {
        removed[j] = true;
      }
    }
  }
  return kept;
}

/// Runs NMS independently per class so a fire box never suppresses a smoke box.
List<Detection> nmsPerClass(
  List<Detection> detections,
  double iouThreshold, {
  int classCount = 2,
}) {
  final result = <Detection>[];
  for (var c = 0; c < classCount; c++) {
    final perClass = detections.where((d) => d.classId == c).toList();
    result.addAll(nonMaxSuppression(perClass, iouThreshold));
  }
  return result;
}
