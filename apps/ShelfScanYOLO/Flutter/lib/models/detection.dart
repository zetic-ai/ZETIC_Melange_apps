import 'dart:math' as math;

/// An axis-aligned bounding box in *original-image pixel* space
/// (x1,y1 = top-left, x2,y2 = bottom-right).
///
/// Coordinates are pixels, NOT normalized 0..1 — the model emits boxes in
/// 640x640 letterbox pixel space (raw channels reach ~640) which the
/// postprocessor inverts into original-image pixels.
class BBox {
  const BBox(this.x1, this.y1, this.x2, this.y2);

  final double x1;
  final double y1;
  final double x2;
  final double y2;

  double get width => math.max(0.0, x2 - x1);
  double get height => math.max(0.0, y2 - y1);
  double get area => width * height;

  /// Intersection-over-union with [other]. Matches the reference pipeline's
  /// clamped-area / `+1e-9` denominator (validate_demo.py) exactly.
  double iou(BBox other) {
    final ix1 = math.max(x1, other.x1);
    final iy1 = math.max(y1, other.y1);
    final ix2 = math.min(x2, other.x2);
    final iy2 = math.min(y2, other.y2);
    final iw = math.max(0.0, ix2 - ix1);
    final ih = math.max(0.0, iy2 - iy1);
    final inter = iw * ih;
    return inter / (area + other.area - inter + 1e-9);
  }

  @override
  String toString() =>
      'BBox(${x1.toStringAsFixed(1)}, ${y1.toStringAsFixed(1)}, '
      '${x2.toStringAsFixed(1)}, ${y2.toStringAsFixed(1)})';
}

/// One detected product facing / SKU: a box plus its (already-sigmoid'd)
/// class confidence. Single-class model, so the label is always "product".
class Detection {
  const Detection(this.box, this.confidence);

  final BBox box;
  final double confidence;

  static const String label = 'product';

  @override
  String toString() =>
      'Detection($box, conf=${confidence.toStringAsFixed(3)})';
}
