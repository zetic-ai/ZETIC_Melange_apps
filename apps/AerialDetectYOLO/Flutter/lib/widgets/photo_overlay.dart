import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../models/label.dart';
import 'coordinate_mapping.dart';

/// Paints a still photo + its detection boxes under a single BoxFit.contain
/// transform, so the drawn image and the overlaid boxes cannot drift apart.
///
/// Detections are in the still's source-pixel space ([Size(image.width,
/// image.height)]); both the image and every box go through [computeContainFit]
/// / [mapContainRect] with that same size. Chip styling matches the live
/// [DetectionOverlay].
class PhotoOverlay extends StatelessWidget {
  const PhotoOverlay({
    super.key,
    required this.image,
    required this.detections,
  });

  final ui.Image image;
  final List<Detection> detections;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _PhotoPainter(image, detections),
      size: Size.infinite,
    );
  }
}

class _PhotoPainter extends CustomPainter {
  _PhotoPainter(this.image, this.detections);

  final ui.Image image;
  final List<Detection> detections;

  @override
  void paint(Canvas canvas, Size size) {
    if (size.width <= 0 || size.height <= 0) return;
    final Size imageSize =
        Size(image.width.toDouble(), image.height.toDouble());

    // 1) The still, letterboxed into the canvas.
    final ContainFit fit = computeContainFit(imageSize, size);
    canvas.drawImageRect(
      image,
      Rect.fromLTWH(0, 0, imageSize.width, imageSize.height),
      fit.destRect,
      Paint()..filterQuality = FilterQuality.medium,
    );

    // 2) Boxes, using the SAME contain transform.
    for (final Detection d in detections) {
      final Color color = colorForClass(d.classId);
      final Rect rect = mapContainRect(d.rect, imageSize, size);

      canvas.drawRect(
        rect,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.5
          ..color = color,
      );

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
  bool shouldRepaint(_PhotoPainter old) =>
      !identical(old.image, image) ||
      !identical(old.detections, detections);
}
