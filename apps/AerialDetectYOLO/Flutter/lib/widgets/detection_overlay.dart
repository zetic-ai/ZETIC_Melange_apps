import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../models/label.dart';
import 'coordinate_mapping.dart';

/// Paints detection boxes over the camera preview.
///
/// Detections are in source-buffer pixel space ([imageSize]). They are mapped to
/// the canvas with BoxFit.cover — the SAME fit the preview uses — and with NO
/// rotation/transpose (the PyroGuard lesson: the buffer arrives upright, so a
/// spurious 90° rotation is the bug, not a fix). If a device delivers a rotated
/// buffer, fix it here against the on-HUD buffer WxH readout.
class DetectionOverlay extends StatelessWidget {
  const DetectionOverlay({
    super.key,
    required this.detections,
    required this.imageSize,
  });

  final List<Detection> detections;
  final Size imageSize;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _DetectionPainter(detections, imageSize),
      size: Size.infinite,
    );
  }
}

class _DetectionPainter extends CustomPainter {
  _DetectionPainter(this.detections, this.imageSize);

  final List<Detection> detections;
  final Size imageSize;

  @override
  void paint(Canvas canvas, Size size) {
    if (imageSize.width <= 0 || imageSize.height <= 0) return;

    for (final Detection d in detections) {
      final Color color = colorForClass(d.classId);
      final Rect rect = mapCoverRect(d.rect, imageSize, size);

      final Paint box = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5
        ..color = color;
      canvas.drawRect(rect, box);

      // Label chip.
      final TextPainter tp = TextPainter(
        text: TextSpan(
          text: ' ${d.label} ${(d.confidence * 100).toStringAsFixed(0)}% ',
          style: const TextStyle(
            color: Colors.black,
            fontSize: 11,
            fontWeight: FontWeight.w600,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      final Rect chip = Rect.fromLTWH(
        rect.left,
        (rect.top - tp.height - 2).clamp(0.0, size.height),
        tp.width,
        tp.height + 2,
      );
      canvas.drawRect(chip, Paint()..color = color);
      tp.paint(canvas, Offset(chip.left, chip.top + 1));
    }
  }

  @override
  bool shouldRepaint(_DetectionPainter old) =>
      !identical(old.detections, detections) || old.imageSize != imageSize;
}
