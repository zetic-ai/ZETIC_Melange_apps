import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../services/display_fit.dart';
import '../theme.dart';

/// Shows the uploaded image (BoxFit.contain) with detection boxes drawn on top,
/// mapped from original-image px onto the displayed-image rect. Image + overlay
/// share the same constraints via a single LayoutBuilder so the fit transform
/// used to draw matches the fit used to display.
class DetectionOverlay extends StatelessWidget {
  const DetectionOverlay({
    super.key,
    required this.imageBytes,
    required this.imageWidth,
    required this.imageHeight,
    required this.detections,
    this.showConfidence = true,
  });

  final Uint8List imageBytes;
  final int imageWidth;
  final int imageHeight;
  final List<Detection> detections;
  final bool showConfidence;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final w = constraints.maxWidth;
        final h = constraints.maxHeight;
        return Stack(
          fit: StackFit.expand,
          children: [
            Image.memory(
              imageBytes,
              fit: BoxFit.contain,
              width: w,
              height: h,
              gaplessPlayback: true,
              filterQuality: FilterQuality.medium,
            ),
            CustomPaint(
              size: Size(w, h),
              painter: _BoxPainter(
                detections: detections,
                imageWidth: imageWidth,
                imageHeight: imageHeight,
                showConfidence: showConfidence,
              ),
            ),
          ],
        );
      },
    );
  }
}

class _BoxPainter extends CustomPainter {
  _BoxPainter({
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
    required this.showConfidence,
  });

  final List<Detection> detections;
  final int imageWidth;
  final int imageHeight;
  final bool showConfidence;

  @override
  void paint(Canvas canvas, Size size) {
    if (imageWidth <= 0 || imageHeight <= 0) return;
    final fit = DisplayFit.contain(
      imageWidth: imageWidth,
      imageHeight: imageHeight,
      widgetWidth: size.width,
      widgetHeight: size.height,
    );

    // Thin stroke so hundreds of dense boxes stay legible.
    final stroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = detections.length > 250 ? 1.0 : 1.6
      ..color = ShelfSenseTheme.accent;

    final showLabels = showConfidence && detections.length <= 40;

    for (final d in detections) {
      final r = fit.mapBox(d.box);
      final rect = Rect.fromLTRB(r[0], r[1], r[2], r[3]);
      canvas.drawRect(rect, stroke);
      if (showLabels) {
        _drawLabel(canvas, rect, '${(d.confidence * 100).round()}%');
      }
    }
  }

  void _drawLabel(Canvas canvas, Rect rect, String text) {
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: const TextStyle(
          color: Colors.black,
          fontSize: 10,
          fontWeight: FontWeight.w700,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    final bg = Rect.fromLTWH(
      rect.left,
      (rect.top - tp.height - 2).clamp(0.0, double.infinity),
      tp.width + 6,
      tp.height + 2,
    );
    canvas.drawRect(bg, Paint()..color = ShelfSenseTheme.accent);
    tp.paint(canvas, Offset(bg.left + 3, bg.top + 1));
  }

  @override
  bool shouldRepaint(_BoxPainter old) =>
      old.detections != detections ||
      old.imageWidth != imageWidth ||
      old.imageHeight != imageHeight ||
      old.showConfidence != showConfidence;
}
