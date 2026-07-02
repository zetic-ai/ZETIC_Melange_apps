import 'dart:math' as math;

import '../models/detection.dart';

/// The forward letterbox parameters recorded during preprocessing, and the
/// inverse used during postprocessing. Kept as a value object so the SAME
/// numbers used to letterbox the image are used to un-letterbox the boxes.
///
/// Matches validate_demo.py exactly:
///   scale = min(target/W, target/H)
///   resizedW = round(W*scale), resizedH = round(H*scale)
///   padX = (target - resizedW) ~/ 2   (integer floor division)
///   padY = (target - resizedH) ~/ 2
class LetterboxTransform {
  const LetterboxTransform._({
    required this.originalWidth,
    required this.originalHeight,
    required this.targetSize,
    required this.scale,
    required this.resizedWidth,
    required this.resizedHeight,
    required this.padX,
    required this.padY,
  });

  final int originalWidth;
  final int originalHeight;
  final int targetSize;
  final double scale;
  final int resizedWidth;
  final int resizedHeight;
  final int padX;
  final int padY;

  factory LetterboxTransform.compute({
    required int originalWidth,
    required int originalHeight,
    int targetSize = 640,
  }) {
    assert(originalWidth > 0 && originalHeight > 0, 'image dims must be > 0');
    final scale =
        math.min(targetSize / originalWidth, targetSize / originalHeight);
    final resizedWidth = (originalWidth * scale).round();
    final resizedHeight = (originalHeight * scale).round();
    final padX = (targetSize - resizedWidth) ~/ 2;
    final padY = (targetSize - resizedHeight) ~/ 2;
    return LetterboxTransform._(
      originalWidth: originalWidth,
      originalHeight: originalHeight,
      targetSize: targetSize,
      scale: scale,
      resizedWidth: resizedWidth,
      resizedHeight: resizedHeight,
      padX: padX,
      padY: padY,
    );
  }

  // ---- forward: original-image px -> 640 letterbox px ----
  double toLetterboxX(double x) => x * scale + padX;
  double toLetterboxY(double y) => y * scale + padY;

  // ---- inverse: 640 letterbox px -> original-image px ----
  double toOriginalX(double x) => (x - padX) / scale;
  double toOriginalY(double y) => (y - padY) / scale;

  /// Invert a box from 640 letterbox pixel space back to original-image px.
  BBox letterboxToOriginal(BBox b) => BBox(
        toOriginalX(b.x1),
        toOriginalY(b.y1),
        toOriginalX(b.x2),
        toOriginalY(b.y2),
      );
}
