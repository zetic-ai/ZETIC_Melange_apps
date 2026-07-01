import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../services/fit.dart';
import '../theme.dart';

/// Paints plate boxes over the image, mapping upright image-space boxes to the
/// screen with the SAME BoxFit transform the image is displayed under: cover
/// for the live camera preview (fills the screen), contain for the still photo
/// (letterboxed). No rotation is applied: detections are already in upright
/// image space (the reference iOS buffer is upright; the bug to avoid is a
/// *spurious* rotation).
class DetectionOverlay extends StatelessWidget {
  const DetectionOverlay({
    super.key,
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
    this.plateOf,
    this.revision = 0,
    this.cover = true,
  });

  final List<Detection> detections;
  final int imageWidth;
  final int imageHeight;

  /// Resolver for the recognized plate string of a detection (null if not yet
  /// read). Lets the box render its OCR'd text while staying detection-only.
  final String? Function(Detection)? plateOf;

  /// Bumped by the OCR pipeline so the painter repaints when plate text updates
  /// even though [detections] is the same instance.
  final int revision;

  /// True => BoxFit.cover (live preview). False => BoxFit.contain (still photo).
  final bool cover;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _OverlayPainter(
        detections,
        imageWidth,
        imageHeight,
        plateOf,
        revision,
        cover,
      ),
      size: Size.infinite,
    );
  }
}

class _OverlayPainter extends CustomPainter {
  _OverlayPainter(
    this.detections,
    this.imageWidth,
    this.imageHeight,
    this.plateOf,
    this.revision,
    this.cover,
  );

  final List<Detection> detections;
  final int imageWidth;
  final int imageHeight;
  final String? Function(Detection)? plateOf;
  final int revision;
  final bool cover;

  @override
  void paint(Canvas canvas, Size size) {
    if (imageWidth <= 0 || imageHeight <= 0 || detections.isEmpty) return;

    // Map image-space -> screen with the same fit the image is displayed under.
    final map = cover
        ? FitMapping.cover(imageWidth.toDouble(), imageHeight.toDouble(),
            size.width, size.height)
        : FitMapping.contain(imageWidth.toDouble(), imageHeight.toDouble(),
            size.width, size.height);
    final scale = map.scale;
    final dx = map.dx;
    final dy = map.dy;

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
      _label(
        canvas,
        rect,
        'plate ${(d.confidence * 100).toStringAsFixed(0)}%',
        atTop: true,
      );
      // Recognized plate text (on-device OCR), shown under the box when read.
      final plate = plateOf?.call(d);
      if (plate != null && plate.isNotEmpty) {
        _label(canvas, rect, plate, atTop: false);
      }
    }
  }

  void _label(Canvas canvas, Rect rect, String text, {required bool atTop}) {
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
    final top = atTop
        ? math.max(0.0, rect.top - tp.height - 4)
        : rect.bottom + 2;
    final bg = Rect.fromLTWH(rect.left, top, tp.width + 10, tp.height + 4);
    canvas.drawRect(bg, Paint()..color = AppTheme.accent);
    tp.paint(canvas, Offset(bg.left + 5, bg.top + 2));
  }

  @override
  bool shouldRepaint(_OverlayPainter old) =>
      !identical(old.detections, detections) ||
      old.imageWidth != imageWidth ||
      old.imageHeight != imageHeight ||
      old.revision != revision ||
      old.cover != cover;
}
