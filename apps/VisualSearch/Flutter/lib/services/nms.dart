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

/// GLOBAL (class-agnostic) non-maximum suppression.
///
/// SnapSeek uses the detector as a salient-object localizer, not a classifier:
/// two overlapping boxes are the SAME object regardless of their COCO label
/// (e.g. "person" vs "handbag" on the same figure), so a class-agnostic pass is
/// correct here — NOT per-class. Returns the survivors sorted by descending
/// confidence, so `result.first` is the primary box.
List<Detection> globalNms(List<Detection> detections, double iouThreshold) {
  if (detections.length <= 1) return List<Detection>.of(detections);

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
