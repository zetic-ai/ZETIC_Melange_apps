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

/// Greedy non-maximum suppression over a single class's detections.
///
/// Sorts by descending confidence, keeps the strongest box, discards any
/// remaining box whose IoU with a kept box exceeds [iouThreshold]. Box areas
/// are pre-computed once (Tier B: avoids recomputing width*height inside the
/// O(n^2) loop).
List<Detection> nonMaxSuppression(
  List<Detection> detections,
  double iouThreshold,
) {
  if (detections.length <= 1) return detections;

  final sorted = List<Detection>.of(detections)
    ..sort((a, b) => b.confidence.compareTo(a.confidence));
  final int n = sorted.length;
  final areas = List<double>.generate(
    n,
    (i) => sorted[i].rect.width * sorted[i].rect.height,
    growable: false,
  );
  final removed = List<bool>.filled(n, false);
  final kept = <Detection>[];

  for (var i = 0; i < n; i++) {
    if (removed[i]) continue;
    final ri = sorted[i].rect;
    kept.add(sorted[i]);
    for (var j = i + 1; j < n; j++) {
      if (removed[j]) continue;
      final rj = sorted[j].rect;
      final double left = ri.left > rj.left ? ri.left : rj.left;
      final double top = ri.top > rj.top ? ri.top : rj.top;
      final double right = ri.right < rj.right ? ri.right : rj.right;
      final double bottom = ri.bottom < rj.bottom ? ri.bottom : rj.bottom;
      final double iw = right - left;
      final double ih = bottom - top;
      if (iw <= 0 || ih <= 0) continue;
      final double inter = iw * ih;
      final double union = areas[i] + areas[j] - inter;
      if (union > 0 && inter / union > iouThreshold) {
        removed[j] = true;
      }
    }
  }
  return kept;
}

/// Runs NMS independently PER CLASS (never globally), so an overlapping
/// Hardhat box and Safety-Vest box on the same worker both survive.
/// Detections are bucketed by classId in one pass.
List<Detection> nmsPerClass(List<Detection> detections, double iouThreshold) {
  if (detections.length <= 1) return detections;

  final buckets = <int, List<Detection>>{};
  for (final d in detections) {
    (buckets[d.classId] ??= <Detection>[]).add(d);
  }
  final result = <Detection>[];
  for (final perClass in buckets.values) {
    result.addAll(nonMaxSuppression(perClass, iouThreshold));
  }
  return result;
}
