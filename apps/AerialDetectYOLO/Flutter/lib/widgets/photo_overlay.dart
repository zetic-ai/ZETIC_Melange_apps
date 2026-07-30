import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../models/label.dart';
import 'coordinate_mapping.dart';

/// Paints a still photo + its detection boxes under a single BoxFit.contain
/// transform, so the drawn image and the overlaid boxes cannot drift apart.
///
/// Legibility-first (user feedback: heavy per-box fill stacked into solid green
/// blobs over the cars when zoomed in): boxes are drawn as crisp, class-colored
/// OUTLINE-ONLY strokes with NO fill and NO per-box confidence text. Only the
/// [labelTopN] highest-confidence boxes get a small label chip, so the
/// underlying cars stay clearly visible. Per-class counts live in the top bar,
/// not on the image.
class PhotoOverlay extends StatelessWidget {
  const PhotoOverlay({
    super.key,
    required this.image,
    required this.detections,
    this.labelTopN = 3,
    this.viewScale = 1.0,
  });

  final ui.Image image;
  final List<Detection> detections;

  /// How many of the highest-confidence boxes get a label chip (0 = none).
  final int labelTopN;

  /// Current InteractiveViewer zoom (getMaxScaleOnAxis). The whole canvas is
  /// scaled by the viewer, so box strokes and label text are divided by this so
  /// they stay a CONSTANT thin thickness/size on screen at every zoom level
  /// (otherwise a 2px stroke becomes ~16px at 8x and re-buries the cars).
  final double viewScale;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _PhotoPainter(image, detections, labelTopN, viewScale),
      size: Size.infinite,
    );
  }
}

class _PhotoPainter extends CustomPainter {
  _PhotoPainter(this.image, this.detections, this.labelTopN, this.viewScale);

  /// On-screen box-stroke thickness in logical px, kept constant across zoom.
  static const double kBaseStrokeWidth = 2.0;

  /// On-screen label font size in logical px, kept constant across zoom.
  static const double kBaseLabelFontSize = 11.0;

  final ui.Image image;
  final List<Detection> detections;
  final int labelTopN;
  final double viewScale;

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

    // Divide by the current zoom so on-screen thickness stays constant: the
    // viewer scales the canvas by `viewScale`, so a stroke drawn at
    // base/viewScale in image space renders at ~base logical px on screen.
    // Clamp the image-space width so it never vanishes (min 0.3) and is never
    // fatter than the 1x base.
    final double scale = viewScale <= 0 ? 1.0 : viewScale;
    final double strokeWidth =
        (kBaseStrokeWidth / scale).clamp(0.3, kBaseStrokeWidth);

    // 2) Outline-only, class-colored boxes — NO fill. A crisp opaque stroke
    // means tightly-packed boxes never stack into solid blobs, so every car
    // stays visible even when zoomed in.
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
    }

    // 3) Label ONLY the top-N highest-confidence boxes (drawn last, on top).
    if (labelTopN > 0 && detections.isNotEmpty) {
      final List<Detection> top = List<Detection>.of(detections)
        ..sort((Detection a, Detection b) =>
            b.confidence.compareTo(a.confidence));
      final int n = top.length < labelTopN ? top.length : labelTopN;
      // Font + gaps in image space also divided by zoom, so the chips stay a
      // constant readable size on screen instead of ballooning when zoomed.
      final double fontSize = kBaseLabelFontSize / scale;
      final double gap = 2.0 / scale;
      for (int i = 0; i < n; i++) {
        final Detection d = top[i];
        final Color color = colorForClass(d.classId);
        final Rect rect = mapContainRect(d.rect, imageSize, size);

        final TextPainter tp = TextPainter(
          text: TextSpan(
            text: ' ${d.label} ${(d.confidence * 100).toStringAsFixed(0)}% ',
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
  }

  @override
  bool shouldRepaint(_PhotoPainter old) =>
      !identical(old.image, image) ||
      !identical(old.detections, detections) ||
      old.labelTopN != labelTopN ||
      old.viewScale != viewScale;
}
