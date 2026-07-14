import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../services/fit.dart';
import '../theme.dart';

/// Paints thin plate boxes over the uploaded still, mapping upright image-space
/// boxes to the screen with the SAME BoxFit.contain transform the image is
/// displayed under (letterboxed). Each box carries a small index badge (1, 2, …)
/// that matches its row in the results list below, so the user can link a box on
/// the image to its zoomed crop + OCR text. No fat labels — the reading happens
/// in the list, so the image stays clean and un-cluttered.
class DetectionOverlay extends StatelessWidget {
  const DetectionOverlay({
    super.key,
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
  });

  final List<Detection> detections;
  final int imageWidth;
  final int imageHeight;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _OverlayPainter(detections, imageWidth, imageHeight),
      size: Size.infinite,
    );
  }
}

class _OverlayPainter extends CustomPainter {
  _OverlayPainter(this.detections, this.imageWidth, this.imageHeight);

  final List<Detection> detections;
  final int imageWidth;
  final int imageHeight;

  @override
  void paint(Canvas canvas, Size size) {
    if (imageWidth <= 0 || imageHeight <= 0 || detections.isEmpty) return;

    final map = FitMapping.contain(imageWidth.toDouble(), imageHeight.toDouble(),
        size.width, size.height);
    final scale = map.scale;
    final dx = map.dx;
    final dy = map.dy;

    final box = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = AppTheme.accent;

    for (var i = 0; i < detections.length; i++) {
      final d = detections[i];
      final rect = Rect.fromLTRB(
        d.left * scale + dx,
        d.top * scale + dy,
        d.right * scale + dx,
        d.bottom * scale + dy,
      );
      canvas.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(3)),
        box,
      );
      _badge(canvas, rect, '${i + 1}');
    }
  }

  /// Small numbered badge at the box's top-left corner, tying it to the list.
  void _badge(Canvas canvas, Rect rect, String text) {
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: const TextStyle(
          color: AppTheme.bg,
          fontSize: 11,
          fontWeight: FontWeight.w800,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    const pad = 4.0;
    final w = tp.width + pad * 2;
    final h = tp.height + pad;
    final r = Rect.fromLTWH(rect.left, rect.top - h, w, h);
    canvas.drawRRect(
      RRect.fromRectAndCorners(
        r,
        topLeft: const Radius.circular(3),
        topRight: const Radius.circular(3),
      ),
      Paint()..color = AppTheme.accent,
    );
    tp.paint(canvas, Offset(r.left + pad, r.top + pad / 2));
  }

  @override
  bool shouldRepaint(_OverlayPainter old) =>
      !identical(old.detections, detections) ||
      old.imageWidth != imageWidth ||
      old.imageHeight != imageHeight;
}
