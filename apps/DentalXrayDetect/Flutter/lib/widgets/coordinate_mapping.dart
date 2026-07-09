import 'dart:ui';

/// Placement of an [imageSize] image inside [canvasSize] under BoxFit.contain:
/// the uniform scale plus the top-left offset of the fitted image. Shared by the
/// still-photo painter so the drawn image and the boxes use ONE transform (a
/// mismatch is the classic overlay-drift bug). This composes with the letterbox
/// inverse: detections are already in original-image pixel space, and this maps
/// that space to the on-screen (possibly fit-scaled) radiograph.
class ContainFit {
  const ContainFit({
    required this.scale,
    required this.dx,
    required this.dy,
    required this.dispW,
    required this.dispH,
  });

  final double scale;
  final double dx;
  final double dy;
  final double dispW;
  final double dispH;

  /// The destination rect the whole image is painted into.
  Rect get destRect => Rect.fromLTWH(dx, dy, dispW, dispH);
}

/// BoxFit.contain scale factor (smaller axis ratio, so the whole image fits).
double containScale(Size imageSize, Size canvasSize) {
  final double sx = canvasSize.width / imageSize.width;
  final double sy = canvasSize.height / imageSize.height;
  return sx < sy ? sx : sy;
}

/// Compute the BoxFit.contain placement of [imageSize] within [canvasSize].
ContainFit computeContainFit(Size imageSize, Size canvasSize) {
  final double scale = containScale(imageSize, canvasSize);
  final double dispW = imageSize.width * scale;
  final double dispH = imageSize.height * scale;
  return ContainFit(
    scale: scale,
    dx: (canvasSize.width - dispW) / 2.0,
    dy: (canvasSize.height - dispH) / 2.0,
    dispW: dispW,
    dispH: dispH,
  );
}

/// Map a rect in image-pixel space to canvas space under BoxFit.contain, with
/// NO rotation (a still radiograph has a static rect — no camera-buffer
/// transpose). Round-trip tested against [unmapContainRect].
Rect mapContainRect(Rect src, Size imageSize, Size canvasSize) {
  final ContainFit f = computeContainFit(imageSize, canvasSize);
  return Rect.fromLTRB(
    f.dx + src.left * f.scale,
    f.dy + src.top * f.scale,
    f.dx + src.right * f.scale,
    f.dy + src.bottom * f.scale,
  );
}

/// Exact inverse of [mapContainRect]: canvas space back to image-pixel space.
Rect unmapContainRect(Rect canvasRect, Size imageSize, Size canvasSize) {
  final ContainFit f = computeContainFit(imageSize, canvasSize);
  return Rect.fromLTRB(
    (canvasRect.left - f.dx) / f.scale,
    (canvasRect.top - f.dy) / f.scale,
    (canvasRect.right - f.dx) / f.scale,
    (canvasRect.bottom - f.dy) / f.scale,
  );
}
