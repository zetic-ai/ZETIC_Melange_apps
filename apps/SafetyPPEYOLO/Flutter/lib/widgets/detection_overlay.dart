import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../theme.dart';
import 'overlay_geometry.dart';

/// Draws PPE bounding boxes + label chips on top of the camera preview.
///
/// Detection rects are normalized 0..1 in the upright camera-frame space; the
/// painter only applies the BoxFit.cover transform (see overlay_geometry.dart
/// for why no rotation happens here).
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

    final double contentAspect = uprightContentAspect(
      previewWidth: imageWidth,
      previewHeight: imageHeight,
      sensorOrientation: sensorOrientation,
    );

    for (final det in detections) {
      final Rect mapped = mapRectCover(det.rect, contentAspect, size);
      _drawBox(canvas, mapped, det);
    }
  }

  void _drawBox(Canvas canvas, Rect rect, Detection det) {
    final Color color = SiteColors.forClass(det.classId);
    final double strokeWidth = det.isViolation ? 4.0 : 2.5;

    final border = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..color = color;
    final rrect = RRect.fromRectAndRadius(rect, const Radius.circular(6));
    canvas.drawRRect(rrect, border);

    // Label chip: "NO HARDHAT 87%".
    final pct = (det.confidence * 100).clamp(0, 100).toStringAsFixed(0);
    final tp = TextPainter(
      text: TextSpan(
        text: ' ${det.label} $pct% ',
        style: TextStyle(
          color: det.isViolation ? Colors.white : Colors.black,
          fontSize: 13,
          fontWeight: FontWeight.w800,
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
    // Repaint only when detections (or geometry) actually change (Tier B).
    return old.detections != detections ||
        old.imageWidth != imageWidth ||
        old.imageHeight != imageHeight ||
        old.sensorOrientation != sensorOrientation;
  }
}
