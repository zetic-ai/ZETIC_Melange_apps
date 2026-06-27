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
