import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../theme.dart';

/// Draws bounding boxes + labels on top of the camera preview.
///
/// Detection rects are normalized 0..1 in the *original camera frame* space
/// (which is in sensor orientation, usually landscape). This painter rotates
/// those coordinates by [sensorOrientation] to match the upright preview, then
/// maps them with a BoxFit.cover transform so they line up with the
/// cover-scaled [CameraPreview].
class DetectionOverlay extends CustomPainter {
  DetectionOverlay({
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
    required this.sensorOrientation,
  });

  final List<Detection> detections;
  final int imageWidth;
  final int imageHeight;
  final int sensorOrientation;

  @override
  void paint(Canvas canvas, Size size) {
    if (detections.isEmpty) return;

    final bool rotated =
        sensorOrientation == 90 || sensorOrientation == 270;

    // Aspect ratio of the *displayed* (upright) frame.
    final double contentW =
        rotated ? imageHeight.toDouble() : imageWidth.toDouble();
    final double contentH =
        rotated ? imageWidth.toDouble() : imageHeight.toDouble();
    final double contentAspect = contentW / contentH;
    final double widgetAspect = size.width / size.height;

    // BoxFit.cover: scale to fill, center, crop the overflow.
    double scaledW, scaledH;
    if (widgetAspect > contentAspect) {
      scaledW = size.width;
      scaledH = size.width / contentAspect;
    } else {
      scaledH = size.height;
      scaledW = size.height * contentAspect;
    }
    final double offsetX = (size.width - scaledW) / 2;
    final double offsetY = (size.height - scaledH) / 2;

    for (final det in detections) {
      final Rect mapped = _mapRect(det.rect, offsetX, offsetY, scaledW, scaledH);
      _drawBox(canvas, mapped, det);
    }
  }

  /// Maps a normalized image-space rect to widget pixels, applying rotation
  /// then the cover transform.
  Rect _mapRect(
    Rect r,
    double offsetX,
    double offsetY,
    double scaledW,
    double scaledH,
  ) {
    final c1 = _rotatePoint(r.left, r.top);
    final c2 = _rotatePoint(r.right, r.bottom);
    final double du1 = c1.dx < c2.dx ? c1.dx : c2.dx;
    final double dv1 = c1.dy < c2.dy ? c1.dy : c2.dy;
    final double du2 = c1.dx > c2.dx ? c1.dx : c2.dx;
    final double dv2 = c1.dy > c2.dy ? c1.dy : c2.dy;

    return Rect.fromLTRB(
      offsetX + du1 * scaledW,
      offsetY + dv1 * scaledH,
      offsetX + du2 * scaledW,
      offsetY + dv2 * scaledH,
    );
  }

  /// Maps a normalized point from detection space to upright display space.
  ///
  /// Detections now arrive upright on both platforms: iOS delivers the buffer
  /// already display-upright, and Android's sensor-orientation buffer is rotated
  /// to upright in [preprocessFrame] before inference. So no rotation is needed
  /// here — the cover-fit aspect handling above (which uses the landscape
  /// previewSize + sensorOrientation to derive the portrait content aspect) does
  /// the rest.
  Offset _rotatePoint(double u, double v) => Offset(u, v);

  void _drawBox(Canvas canvas, Rect rect, Detection det) {
    final Color color = det.isFire ? PyroColors.fire : PyroColors.smoke;
    final double strokeWidth = det.isFire ? 4.0 : 2.5;

    final border = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..color = color;
    final rrect = RRect.fromRectAndRadius(rect, const Radius.circular(6));
    canvas.drawRRect(rrect, border);

    // Label chip: "fire 94%".
    final pct = (det.confidence * 100).clamp(0, 100).toStringAsFixed(0);
    final tp = TextPainter(
      text: TextSpan(
        text: ' ${det.label} $pct% ',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 13,
          fontWeight: FontWeight.w700,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    const double chipH = 20;
    final double chipW = tp.width + 4;
    double chipTop = rect.top - chipH;
    if (chipTop < 0) chipTop = rect.top; // flip inside if off-screen
    final chipRect = Rect.fromLTWH(rect.left, chipTop, chipW, chipH);
    final chipPaint = Paint()..color = color;
    canvas.drawRRect(
      RRect.fromRectAndCorners(
        chipRect,
        topLeft: const Radius.circular(6),
        topRight: const Radius.circular(6),
        bottomRight: const Radius.circular(6),
      ),
      chipPaint,
    );
    tp.paint(canvas, Offset(rect.left + 2, chipTop + (chipH - tp.height) / 2));
  }

  @override
  bool shouldRepaint(covariant DetectionOverlay old) {
    return old.detections != detections ||
        old.imageWidth != imageWidth ||
        old.imageHeight != imageHeight ||
        old.sensorOrientation != sensorOrientation;
  }
}
