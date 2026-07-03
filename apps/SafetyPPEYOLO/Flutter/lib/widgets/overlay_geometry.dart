import 'dart:ui' show Rect, Size;

/// Pure geometry for mapping a normalized (0..1, upright-frame) detection rect
/// onto the widget space of a BoxFit.cover-scaled camera preview.
///
/// Kept free of Flutter widgets so the Tier A orientation test can assert the
/// exact transform round-trips a known box.
///
/// [contentAspect] is width/height of the UPRIGHT displayed frame. The camera
/// plugin reports previewSize in sensor orientation (landscape, long side
/// first); when [sensorOrientation] is 90/270 the aspect must be inverted —
/// that is the ONLY orientation handling the overlay does. Detections arrive
/// already upright (iOS buffer is upright; Android is rotated upright during
/// preprocessing), so adding any box rotation here would be the PyroGuard
/// spurious-rotation bug all over again.
double uprightContentAspect({
  required int previewWidth,
  required int previewHeight,
  required int sensorOrientation,
}) {
  final bool rotated = sensorOrientation == 90 || sensorOrientation == 270;
  final double w = rotated ? previewHeight.toDouble() : previewWidth.toDouble();
  final double h = rotated ? previewWidth.toDouble() : previewHeight.toDouble();
  return w / h;
}

/// Maps normalized rect [r] into widget pixels under BoxFit.cover.
Rect mapRectCover(Rect r, double contentAspect, Size widgetSize) {
  final double widgetAspect = widgetSize.width / widgetSize.height;

  double scaledW, scaledH;
  if (widgetAspect > contentAspect) {
    scaledW = widgetSize.width;
    scaledH = widgetSize.width / contentAspect;
  } else {
    scaledH = widgetSize.height;
    scaledW = widgetSize.height * contentAspect;
  }
  final double offsetX = (widgetSize.width - scaledW) / 2;
  final double offsetY = (widgetSize.height - scaledH) / 2;

  return Rect.fromLTRB(
    offsetX + r.left * scaledW,
    offsetY + r.top * scaledH,
    offsetX + r.right * scaledW,
    offsetY + r.bottom * scaledH,
  );
}

/// Inverse of [mapRectCover] — widget pixels back to normalized content
/// coordinates. Exists for the Tier A round-trip test.
Rect unmapRectCover(Rect widgetRect, double contentAspect, Size widgetSize) {
  final double widgetAspect = widgetSize.width / widgetSize.height;

  double scaledW, scaledH;
  if (widgetAspect > contentAspect) {
    scaledW = widgetSize.width;
    scaledH = widgetSize.width / contentAspect;
  } else {
    scaledH = widgetSize.height;
    scaledW = widgetSize.height * contentAspect;
  }
  final double offsetX = (widgetSize.width - scaledW) / 2;
  final double offsetY = (widgetSize.height - scaledH) / 2;

  return Rect.fromLTRB(
    (widgetRect.left - offsetX) / scaledW,
    (widgetRect.top - offsetY) / scaledH,
    (widgetRect.right - offsetX) / scaledW,
    (widgetRect.bottom - offsetY) / scaledH,
  );
}
