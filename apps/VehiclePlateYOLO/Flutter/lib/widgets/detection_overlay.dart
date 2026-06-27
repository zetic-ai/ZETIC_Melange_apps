import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../theme.dart';

/// Paints plate boxes over the preview, mapping upright image-space boxes to
/// the screen with the SAME BoxFit.cover transform the preview uses. No
/// rotation is applied: detections are already in upright image space (the
/// reference iOS buffer is upright; the bug to avoid is a *spurious* rotation).
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

    // BoxFit.cover: scale to fill, center the overflow.
    final scale = math.max(
      size.width / imageWidth,
      size.height / imageHeight,
    );
    final dx = (size.width - imageWidth * scale) / 2.0;
    final dy = (size.height - imageHeight * scale) / 2.0;

    final box = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5
      ..color = AppTheme.accent;
    final fill = Paint()..color = AppTheme.accentSoft;

    for (final d in detections) {
      final rect = Rect.fromLTRB(
        d.left * scale + dx,
        d.top * scale + dy,
        d.right * scale + dx,
        d.bottom * scale + dy,
      );
      canvas.drawRect(rect, fill);
      canvas.drawRect(rect, box);
      _label(canvas, rect, 'plate ${(d.confidence * 100).toStringAsFixed(0)}%');
    }
  }

  void _label(Canvas canvas, Rect rect, String text) {
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: const TextStyle(
          color: AppTheme.bg,
          fontSize: 12,
          fontWeight: FontWeight.w700,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    final bg = Rect.fromLTWH(
      rect.left,
      math.max(0, rect.top - tp.height - 4),
      tp.width + 10,
      tp.height + 4,
    );
    canvas.drawRect(bg, Paint()..color = AppTheme.accent);
    tp.paint(canvas, Offset(bg.left + 5, bg.top + 2));
  }

  @override
  bool shouldRepaint(_OverlayPainter old) =>
      !identical(old.detections, detections) ||
      old.imageWidth != imageWidth ||
      old.imageHeight != imageHeight;
}
