import 'dart:math' as math;

/// Pure geometry for mapping an image-space point onto a display box under a
/// Flutter [BoxFit]. Both the live camera preview (BoxFit.cover) and the
/// still-photo view (BoxFit.contain) map detection boxes with the SAME formula
/// `screen = image * scale + offset`; only the scale rule and the resulting
/// centering offset differ. Extracted (no Flutter import) so the overlay uses
/// exactly the math the tests assert.
class FitMapping {
  const FitMapping(this.scale, this.dx, this.dy);

  /// Uniform image->screen scale factor.
  final double scale;

  /// Screen-space offset of the image origin (letterbox for contain, negative
  /// overflow for cover).
  final double dx;
  final double dy;

  /// BoxFit.cover: scale to FILL the box (max), centering the overflow.
  factory FitMapping.cover(
    double imageW,
    double imageH,
    double boxW,
    double boxH,
  ) => _forScale(math.max(boxW / imageW, boxH / imageH), imageW, imageH, boxW,
      boxH);

  /// BoxFit.contain: scale to FIT inside the box (min), centering the letterbox.
  factory FitMapping.contain(
    double imageW,
    double imageH,
    double boxW,
    double boxH,
  ) => _forScale(math.min(boxW / imageW, boxH / imageH), imageW, imageH, boxW,
      boxH);

  static FitMapping _forScale(
    double scale,
    double imageW,
    double imageH,
    double boxW,
    double boxH,
  ) {
    final dx = (boxW - imageW * scale) / 2.0;
    final dy = (boxH - imageH * scale) / 2.0;
    return FitMapping(scale, dx, dy);
  }

  double mapX(double x) => x * scale + dx;
  double mapY(double y) => y * scale + dy;
}
