import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../theme.dart';

/// Draws the primary salient box (teal, solid) and the embedded crop rect
/// (amber, dashed) over the captured photo. Both are in normalized 0..1 coords,
/// so the painter just scales them by the paint size — the parent uses an
/// [AspectRatio] matching the frame so this maps 1:1 onto the shown image.
class DetectionOverlay extends CustomPainter {
  DetectionOverlay({required this.primary, required this.cropRect});

  final Detection? primary;
  final Rect cropRect; // normalized

  @override
  void paint(Canvas canvas, Size size) {
    Rect denorm(Rect r) => Rect.fromLTRB(
        r.left * size.width,
        r.top * size.height,
        r.right * size.width,
        r.bottom * size.height);

    // Crop rect (amber dashed) — what actually gets embedded.
    final cropPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = SnapColors.amber;
    _drawDashedRect(canvas, denorm(cropRect), cropPaint);

    // Primary box (teal solid) with a confidence chip.
    final p = primary;
    if (p != null) {
      final box = denorm(p.rect);
      final boxPaint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3
        ..color = SnapColors.accent;
      canvas.drawRRect(
          RRect.fromRectAndRadius(box, const Radius.circular(6)), boxPaint);

      final tp = TextPainter(
        text: TextSpan(
          text: ' ${p.label}  ${(p.confidence * 100).toStringAsFixed(0)}% ',
          style: const TextStyle(
              color: Colors.black,
              fontSize: 12,
              fontWeight: FontWeight.w700),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      final chipRect = Rect.fromLTWH(
          box.left, (box.top - 18).clamp(0.0, size.height), tp.width, 18);
      canvas.drawRect(chipRect, Paint()..color = SnapColors.accent);
      tp.paint(canvas, chipRect.topLeft);
    }
  }

  void _drawDashedRect(Canvas canvas, Rect r, Paint paint) {
    const dash = 8.0, gap = 5.0;
    void line(Offset a, Offset b) {
      final total = (b - a).distance;
      final dir = (b - a) / total;
      double d = 0;
      while (d < total) {
        final end = d + dash < total ? d + dash : total;
        canvas.drawLine(a + dir * d, a + dir * end, paint);
        d += dash + gap;
      }
    }

    line(r.topLeft, r.topRight);
    line(r.topRight, r.bottomRight);
    line(r.bottomRight, r.bottomLeft);
    line(r.bottomLeft, r.topLeft);
  }

  @override
  bool shouldRepaint(covariant DetectionOverlay old) =>
      old.primary != primary || old.cropRect != cropRect;
}
