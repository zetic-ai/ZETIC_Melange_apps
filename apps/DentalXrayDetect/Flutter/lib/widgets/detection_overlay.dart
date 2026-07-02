import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../models/label.dart';
import 'coordinate_mapping.dart';

/// Paints a radiograph + its detection boxes under a single BoxFit.contain
/// transform, so the drawn image and the overlaid boxes cannot drift apart.
///
/// Each detection is a class-colored outline stroke plus a small label chip
/// ("caries 81%"). Stroke/label thickness is divided by the current zoom so it
/// stays a constant readable size on screen at every InteractiveViewer scale.
class DetectionOverlay extends StatelessWidget {
  const DetectionOverlay({
    super.key,
    required this.image,
    required this.detections,
    this.viewScale = 1.0,
  });

  final ui.Image image;
  final List<Detection> detections;

  /// Current InteractiveViewer zoom (getMaxScaleOnAxis).
  final double viewScale;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _OverlayPainter(image, detections, viewScale),
      size: Size.infinite,
    );
  }
}

class _OverlayPainter extends CustomPainter {
  _OverlayPainter(this.image, this.detections, this.viewScale);

  static const double kBaseStrokeWidth = 2.5;
  static const double kBaseLabelFontSize = 12.0;

  final ui.Image image;
  final List<Detection> detections;
  final double viewScale;

  @override
  void paint(Canvas canvas, Size size) {
    if (size.width <= 0 || size.height <= 0) return;
    final Size imageSize =
        Size(image.width.toDouble(), image.height.toDouble());

    // 1) The radiograph, fitted into the canvas.
    final ContainFit fit = computeContainFit(imageSize, size);
    canvas.drawImageRect(
      image,
      Rect.fromLTWH(0, 0, imageSize.width, imageSize.height),
      fit.destRect,
      Paint()..filterQuality = FilterQuality.medium,
    );

    // Keep on-screen stroke thickness constant across zoom.
    final double scale = viewScale <= 0 ? 1.0 : viewScale;
    final double strokeWidth =
        (kBaseStrokeWidth / scale).clamp(0.4, kBaseStrokeWidth);
    final double fontSize = kBaseLabelFontSize / scale;
    final double gap = 2.0 / scale;

    // 2) Class-colored outline boxes + label chips.
    for (final Detection d in detections) {
      final Color color = colorForClass(d.classId);
      final Rect rect = mapContainRect(d.rect, imageSize, size);

      canvas.drawRect(
        rect,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = strokeWidth
          ..color = color,
      );

      final TextPainter tp = TextPainter(
        text: TextSpan(
          text: ' ${prettyLabel(d.classId)} '
              '${(d.confidence * 100).toStringAsFixed(0)}% ',
          style: TextStyle(
            color: Colors.black,
            fontSize: fontSize,
            fontWeight: FontWeight.w700,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      final Rect chip = Rect.fromLTWH(
        rect.left,
        (rect.top - tp.height - gap).clamp(0.0, size.height),
        tp.width,
        tp.height + gap,
      );
      canvas.drawRect(chip, Paint()..color = color);
      tp.paint(canvas, Offset(chip.left, chip.top + gap / 2));
    }
  }

  @override
  bool shouldRepaint(_OverlayPainter old) =>
      !identical(old.image, image) ||
      !identical(old.detections, detections) ||
      old.viewScale != viewScale;
}
