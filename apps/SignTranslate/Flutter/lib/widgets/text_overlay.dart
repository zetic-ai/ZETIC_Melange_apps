import 'package:flutter/material.dart';

import '../models/text_region.dart';
import '../services/coordinate_mapper.dart';
import '../theme.dart';

/// Draws each recognized region: the quad outline pinned to the sign plus a
/// label chip with the decoded string and confidence.
///
/// Regions arrive in upright-frame space and are mapped to screen space here
/// (BoxFit.cover, matching the preview transform). Repaints ONLY when the
/// result set changes ([revision] guard), not on every widget rebuild.
class TextOverlayPainter extends CustomPainter {
  TextOverlayPainter({
    required this.regions,
    required this.frameWidth,
    required this.frameHeight,
    required this.revision,
  });

  final List<RecognizedRegion> regions;
  final double frameWidth;
  final double frameHeight;

  /// Monotonic counter bumped by the screen whenever results change.
  final int revision;

  @override
  void paint(Canvas canvas, Size size) {
    if (frameWidth <= 0 || frameHeight <= 0) return;

    final outline = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = GlyphColors.accent;
    final outlineDim = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5
      ..color = GlyphColors.accent.withValues(alpha: 0.45);
    final fill = Paint()
      ..style = PaintingStyle.fill
      ..color = GlyphColors.accent.withValues(alpha: 0.08);

    for (final region in regions) {
      final pts = [
        for (final p in region.quad.points)
          mapFrameToScreen(p, frameWidth, frameHeight, size),
      ];
      final path = Path()
        ..moveTo(pts[0].dx, pts[0].dy)
        ..lineTo(pts[1].dx, pts[1].dy)
        ..lineTo(pts[2].dx, pts[2].dy)
        ..lineTo(pts[3].dx, pts[3].dy)
        ..close();

      final hasText = region.text.isNotEmpty;
      canvas.drawPath(path, fill);
      canvas.drawPath(path, hasText ? outline : outlineDim);
      if (hasText) _drawLabel(canvas, size, pts, region);
    }
  }

  void _drawLabel(
    Canvas canvas,
    Size size,
    List<Offset> pts,
    RecognizedRegion region,
  ) {
    final painter = TextPainter(
      text: TextSpan(
        children: [
          TextSpan(
            text: region.text,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 15,
              fontWeight: FontWeight.w700,
            ),
          ),
          TextSpan(
            text: '  ${(region.confidence * 100).round()}%',
            style: TextStyle(
              color: GlyphColors.accent.withValues(alpha: 0.9),
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
      textDirection: TextDirection.ltr,
      maxLines: 1,
      ellipsis: '…',
    )..layout(maxWidth: size.width - 16);

    // Anchor the chip above the quad's top edge, clamped on-screen.
    final anchorX = (pts[0].dx + pts[1].dx) / 2 - painter.width / 2;
    final anchorY = (pts[0].dy + pts[1].dy) / 2 - painter.height - 10;
    final x = anchorX.clamp(4.0, size.width - painter.width - 4.0);
    final y = anchorY.clamp(4.0, size.height - painter.height - 4.0);

    final chip = RRect.fromRectAndRadius(
      Rect.fromLTWH(x - 6, y - 3, painter.width + 12, painter.height + 6),
      const Radius.circular(6),
    );
    canvas.drawRRect(
      chip,
      Paint()..color = GlyphColors.background.withValues(alpha: 0.82),
    );
    canvas.drawRRect(
      chip,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1
        ..color = GlyphColors.accent.withValues(alpha: 0.5),
    );
    painter.paint(canvas, Offset(x, y));
  }

  @override
  bool shouldRepaint(TextOverlayPainter oldDelegate) =>
      oldDelegate.revision != revision;
}
