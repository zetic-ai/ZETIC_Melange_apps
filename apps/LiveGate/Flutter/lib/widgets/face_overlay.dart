import 'package:flutter/material.dart';

import '../models/gate_result.dart';
import '../models/geometry.dart';
import '../theme.dart';

/// Draws the detected face box (color-coded by verdict) over the camera preview.
///
/// The box arrives in upright buffer space ([bufferWidth]x[bufferHeight]); it is
/// mapped to the widget with a BoxFit.cover transform and, for the front camera,
/// mirrored horizontally to match the mirrored self-view preview.
class FaceOverlay extends CustomPainter {
  FaceOverlay({
    required this.box,
    required this.bufferWidth,
    required this.bufferHeight,
    required this.decision,
    required this.mirror,
  });

  final FaceBox? box;
  final int bufferWidth;
  final int bufferHeight;
  final GateDecision decision;
  final bool mirror;

  @override
  void paint(Canvas canvas, Size size) {
    final b = box;
    if (b == null || bufferWidth == 0 || bufferHeight == 0) return;

    // BoxFit.cover mapping from buffer space to the widget.
    final double scale = (size.width / bufferWidth) > (size.height / bufferHeight)
        ? size.width / bufferWidth
        : size.height / bufferHeight;
    final double dx = (size.width - bufferWidth * scale) / 2;
    final double dy = (size.height - bufferHeight * scale) / 2;

    double mapX(double x) {
      final sx = x * scale + dx;
      return mirror ? size.width - sx : sx;
    }

    double mapY(double y) => y * scale + dy;

    final left = mapX(b.x);
    final right = mapX(b.x + b.w);
    final rect = Rect.fromLTRB(
      left < right ? left : right,
      mapY(b.y),
      left < right ? right : left,
      mapY(b.y + b.h),
    );

    final color = switch (decision) {
      GateDecision.pass => GateColors.pass,
      GateDecision.spoof => GateColors.fail,
      GateDecision.noFace => Colors.white24,
      _ => GateColors.warn,
    };

    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = color;
    canvas.drawRRect(
      RRect.fromRectAndRadius(rect, const Radius.circular(14)),
      paint,
    );
  }

  @override
  bool shouldRepaint(covariant FaceOverlay old) =>
      old.box != box ||
      old.decision != decision ||
      old.bufferWidth != bufferWidth ||
      old.bufferHeight != bufferHeight ||
      old.mirror != mirror;
}
