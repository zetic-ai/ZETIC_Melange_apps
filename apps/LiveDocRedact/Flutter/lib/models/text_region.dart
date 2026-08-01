import 'dart:ui';

/// One text region found by the DBNet detector, in upright source-frame
/// pixel coordinates (after the letterbox inverse).
class TextRegion {
  const TextRegion({
    required this.quad,
    required this.bbox,
    required this.score,
  });

  /// Four corners ordered top-left, top-right, bottom-right, bottom-left,
  /// oriented so the text baseline runs left-to-right along quad[0]->quad[1].
  final List<Offset> quad;

  /// Axis-aligned bounding box of [quad].
  final Rect bbox;

  /// Mean heatmap probability over the region's pixels (raw, no extra
  /// activation — the DB head bakes in a Sigmoid).
  final double score;
}
