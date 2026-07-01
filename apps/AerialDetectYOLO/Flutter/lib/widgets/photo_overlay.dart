import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../models/label.dart';
import 'coordinate_mapping.dart';

/// Paints a still photo + its detection boxes under a single BoxFit.contain
/// transform, so the drawn image and the overlaid boxes cannot drift apart.
///
/// Legibility-first (user feedback: ~145 "car 36%" chips buried the cars):
/// boxes are drawn as THIN, class-colored strokes with only a faint fill, and
/// NO per-box confidence text. Only the [labelTopN] highest-confidence boxes get
/// a small label chip, so the underlying cars stay clearly visible. Per-class
/// counts live in the top bar, not on the image.
class PhotoOverlay extends StatelessWidget {
  const PhotoOverlay({
    super.key,
    required this.image,
    required this.detections,
    this.labelTopN = 3,
  });

  final ui.Image image;
  final List<Detection> detections;

  /// How many of the highest-confidence boxes get a label chip (0 = none).
  final int labelTopN;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _PhotoPainter(image, detections, labelTopN),
      size: Size.infinite,
    );
  }
}

class _PhotoPainter extends CustomPainter {
  _PhotoPainter(this.image, this.detections, this.labelTopN);

  final ui.Image image;
  final List<Detection> detections;
  final int labelTopN;

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

    // 2) Thin, class-colored boxes only — the image/cars stay visible.
    for (final Detection d in detections) {
      final Color color = colorForClass(d.classId);
      final Rect rect = mapContainRect(d.rect, imageSize, size);

      // Faint fill so tightly-packed boxes read as regions without hiding cars.
      canvas.drawRect(
        rect,
        Paint()
          ..style = PaintingStyle.fill
          ..color = color.withValues(alpha: 0.08),
      );
      canvas.drawRect(
        rect,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2
          ..color = color.withValues(alpha: 0.9),
      );
    }

    // 3) Label ONLY the top-N highest-confidence boxes (drawn last, on top).
    if (labelTopN > 0 && detections.isNotEmpty) {
      final List<Detection> top = List<Detection>.of(detections)
        ..sort((Detection a, Detection b) =>
            b.confidence.compareTo(a.confidence));
      final int n = top.length < labelTopN ? top.length : labelTopN;
      for (int i = 0; i < n; i++) {
        final Detection d = top[i];
        final Color color = colorForClass(d.classId);
        final Rect rect = mapContainRect(d.rect, imageSize, size);

        final TextPainter tp = TextPainter(
          text: TextSpan(
            text: ' ${d.label} ${(d.confidence * 100).toStringAsFixed(0)}% ',
            style: const TextStyle(
              color: Colors.black,
              fontSize: 11,
              fontWeight: FontWeight.w700,
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
  }

  @override
  bool shouldRepaint(_PhotoPainter old) =>
      !identical(old.image, image) ||
      !identical(old.detections, detections) ||
      old.labelTopN != labelTopN;
}
