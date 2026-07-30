import 'dart:ui';

/// BoxFit.cover scale factor from an image of [imageSize] into [canvasSize].
double coverScale(Size imageSize, Size canvasSize) {
  final double sx = canvasSize.width / imageSize.width;
  final double sy = canvasSize.height / imageSize.height;
  return sx > sy ? sx : sy;
}

/// Map a rect in image-pixel space to canvas space under BoxFit.cover, with NO
/// rotation (the deliberate orientation choice: the camera buffer arrives
/// upright). A spurious transpose here is the classic overlay bug, so this is
/// kept pure and round-trip tested.
Rect mapCoverRect(Rect src, Size imageSize, Size canvasSize) {
  final double scale = coverScale(imageSize, canvasSize);
  final double dispW = imageSize.width * scale;
  final double dispH = imageSize.height * scale;
  final double dx = (canvasSize.width - dispW) / 2.0;
  final double dy = (canvasSize.height - dispH) / 2.0;
  return Rect.fromLTRB(
    dx + src.left * scale,
    dy + src.top * scale,
    dx + src.right * scale,
    dy + src.bottom * scale,
  );
}

/// Exact inverse of [mapCoverRect]: canvas space back to image-pixel space.
Rect unmapCoverRect(Rect canvasRect, Size imageSize, Size canvasSize) {
  final double scale = coverScale(imageSize, canvasSize);
  final double dispW = imageSize.width * scale;
  final double dispH = imageSize.height * scale;
  final double dx = (canvasSize.width - dispW) / 2.0;
  final double dy = (canvasSize.height - dispH) / 2.0;
  return Rect.fromLTRB(
    (canvasRect.left - dx) / scale,
    (canvasRect.top - dy) / scale,
    (canvasRect.right - dx) / scale,
    (canvasRect.bottom - dy) / scale,
  );
}

/// Placement of an [imageSize] image inside [canvasSize] under BoxFit.contain:
/// the uniform scale plus the top-left offset of the letterboxed image. Shared
/// by the still-photo painter so the drawn image and the boxes use ONE
/// transform (a mismatch is the classic overlay-drift bug).
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
/// NO rotation (a still photo has a static rect — no camera-buffer transpose).
/// Used for the photo-upload overlay; round-trip tested against
/// [unmapContainRect].
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
