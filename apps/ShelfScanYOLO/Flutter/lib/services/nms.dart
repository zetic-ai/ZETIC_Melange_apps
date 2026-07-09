import 'dart:typed_data';

import '../models/detection.dart';

/// Single-class **global** non-maximum suppression.
///
/// The model has ONE class ("object"), so global NMS is correct — there is no
/// per-class grouping. A lower-scoring box is suppressed when its IoU with an
/// already-kept box is *strictly greater* than [iouThreshold]. This matches the
/// reference pipeline (validate_demo.py: `order[iou <= iou_thres]` is kept), so
/// a pair at exactly the threshold both survive.
///
/// Performance (Tier-B): the O(n^2) overlap pass runs over flat [Float32List]
/// coordinate/area arrays rather than dereferencing `Detection.box.x1` objects
/// per comparison — on dense shelves (thousands of raw boxes) this is the
/// dominant Dart cost, and the flat layout roughly halves it. Boxes are
/// pre-sorted once (highest confidence first) and areas pre-computed.
List<Detection> nonMaxSuppression(
  List<Detection> detections,
  double iouThreshold,
) {
  final n = detections.length;
  if (n <= 1) return List<Detection>.of(detections);

  // Flatten into typed arrays once (O(n)); the hot loop touches only these.
  final x1 = Float32List(n);
  final y1 = Float32List(n);
  final x2 = Float32List(n);
  final y2 = Float32List(n);
  final area = Float32List(n);
  for (var i = 0; i < n; i++) {
    final b = detections[i].box;
    x1[i] = b.x1;
    y1[i] = b.y1;
    x2[i] = b.x2;
    y2[i] = b.y2;
    area[i] = b.area;
  }

  // Indices sorted by confidence descending.
  final order = List<int>.generate(n, (i) => i)
    ..sort((a, b) => detections[b].confidence.compareTo(detections[a].confidence));

  final suppressed = Uint8List(n); // 0 = alive, 1 = suppressed
  final keep = <Detection>[];

  for (var oi = 0; oi < n; oi++) {
    final i = order[oi];
    if (suppressed[i] != 0) continue;
    keep.add(detections[i]);
    final ix1 = x1[i], iy1 = y1[i], ix2 = x2[i], iy2 = y2[i], ai = area[i];
    for (var oj = oi + 1; oj < n; oj++) {
      final j = order[oj];
      if (suppressed[j] != 0) continue;
      final ox1 = ix1 > x1[j] ? ix1 : x1[j];
      final oy1 = iy1 > y1[j] ? iy1 : y1[j];
      final ox2 = ix2 < x2[j] ? ix2 : x2[j];
      final oy2 = iy2 < y2[j] ? iy2 : y2[j];
      final iw = ox2 - ox1;
      final ih = oy2 - oy1;
      if (iw <= 0 || ih <= 0) continue; // no overlap -> IoU 0
      final inter = iw * ih;
      final iou = inter / (ai + area[j] - inter + 1e-9);
      if (iou > iouThreshold) suppressed[j] = 1;
    }
  }
  return keep;
}
