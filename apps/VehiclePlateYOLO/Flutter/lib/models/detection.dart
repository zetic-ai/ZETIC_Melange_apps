/// One license-plate detection.
///
/// Coordinates are in **upright source-image pixel space** (letterbox already
/// undone), so a downstream painter only has to map image-space -> screen.
/// Plain data so it can cross the inference-isolate boundary cheaply.
class Detection {
  const Detection({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
    required this.confidence,
    this.label = 'license_plate',
  });

  final double left;
  final double top;
  final double right;
  final double bottom;
  final double confidence;
  final String label;

  double get width => right - left;
  double get height => bottom - top;

  @override
  String toString() =>
      'Detection($label ${confidence.toStringAsFixed(2)} '
      '[${left.toStringAsFixed(1)},${top.toStringAsFixed(1)},'
      '${right.toStringAsFixed(1)},${bottom.toStringAsFixed(1)}])';
}
