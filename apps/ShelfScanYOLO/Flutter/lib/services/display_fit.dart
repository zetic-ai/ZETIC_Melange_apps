import 'dart:math' as math;

import '../models/detection.dart';

/// Maps original-image pixel coordinates onto the on-screen displayed-image
/// rect under `BoxFit.contain` (the whole image is shown, letterboxed inside
/// the widget). This is the final coordinate hop:
///   640 letterbox px --(inverse letterbox)--> original px --(this)--> screen.
///
/// A box that is geometrically correct in original px but drawn with the wrong
/// display transform is the classic overlay failure, so this is kept as a small
/// pure-math value object that can be unit-tested independently.
class DisplayFit {
  const DisplayFit._(this.scale, this.dx, this.dy);

  /// original-px -> screen-px multiplier.
  final double scale;

  /// screen-px offset of the displayed image's top-left (letterbox bars).
  final double dx;
  final double dy;

  factory DisplayFit.contain({
    required int imageWidth,
    required int imageHeight,
    required double widgetWidth,
    required double widgetHeight,
  }) {
    final scale = math.min(
      widgetWidth / imageWidth,
      widgetHeight / imageHeight,
    );
    final dx = (widgetWidth - imageWidth * scale) / 2.0;
    final dy = (widgetHeight - imageHeight * scale) / 2.0;
    return DisplayFit._(scale, dx, dy);
  }

  double mapX(double x) => x * scale + dx;
  double mapY(double y) => y * scale + dy;

  /// Map an original-px box to screen-px [left, top, right, bottom].
  List<double> mapBox(BBox b) => [
        mapX(b.x1),
        mapY(b.y1),
        mapX(b.x2),
        mapY(b.y2),
      ];
}
