import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../models/read_field.dart';
import '../theme.dart';

/// How PII fields are hidden in the live preview. Solid bars are the default
/// deterministic path (GATE-2 ruling); flip to [blur] only if the frame
/// budget proves it free on-device.
enum RedactionStyle { bar, blur }

const RedactionStyle kRedactionStyle = RedactionStyle.bar;

/// Immutable per-frame snapshot of one tracked region for painting.
class RegionView {
  const RegionView({
    required this.bbox,
    required this.text,
    required this.piiClass,
    required this.isRead,
  });

  /// Axis-aligned bbox in upright frame pixels.
  final Rect bbox;
  final String? text;
  final PiiClass piiClass;
  final bool isRead;

  bool get isRedacted => isRead && piiClass.isPii;
}

/// Maps upright-frame pixel coordinates to widget coordinates under the same
/// BoxFit.cover transform the camera preview uses. Shared by the painter and
/// the (optional) blur layer so they can never disagree.
class FrameMapper {
  FrameMapper({
    required this.frameWidth,
    required this.frameHeight,
    required Size widgetSize,
  }) {
    final double contentAspect = frameWidth / frameHeight;
    final double widgetAspect = widgetSize.width / widgetSize.height;
    if (widgetAspect > contentAspect) {
      _scaledW = widgetSize.width;
      _scaledH = widgetSize.width / contentAspect;
    } else {
      _scaledH = widgetSize.height;
      _scaledW = widgetSize.height * contentAspect;
    }
    _offsetX = (widgetSize.width - _scaledW) / 2;
    _offsetY = (widgetSize.height - _scaledH) / 2;
  }

  final double frameWidth;
  final double frameHeight;
  late final double _scaledW, _scaledH, _offsetX, _offsetY;

  Rect mapRect(Rect r) => Rect.fromLTRB(
        _offsetX + r.left / frameWidth * _scaledW,
        _offsetY + r.top / frameHeight * _scaledH,
        _offsetX + r.right / frameWidth * _scaledW,
        _offsetY + r.bottom / frameHeight * _scaledH,
      );
}

/// Draws detected text boxes and PII redaction bars over the camera preview.
///
/// - Unread regions: thin teal outline ("text found, reading…").
/// - Read non-PII regions: blue outline + the recognized text above.
/// - Read PII regions: SOLID redaction bar with the class label — the field
///   content is covered in the live preview.
class RedactionOverlay extends CustomPainter {
  RedactionOverlay({
    required this.regions,
    required this.frameWidth,
    required this.frameHeight,
  });

  final List<RegionView> regions;
  final int frameWidth;
  final int frameHeight;

  @override
  void paint(Canvas canvas, Size size) {
    if (regions.isEmpty || frameWidth <= 0 || frameHeight <= 0) return;
    final mapper = FrameMapper(
      frameWidth: frameWidth.toDouble(),
      frameHeight: frameHeight.toDouble(),
      widgetSize: size,
    );

    for (final region in regions) {
      final rect = mapper.mapRect(region.bbox).inflate(1.5);
      if (region.isRedacted) {
        _drawRedactionBar(canvas, rect, region.piiClass);
      } else {
        _drawTextBox(canvas, rect, region);
      }
    }
  }

  void _drawRedactionBar(Canvas canvas, Rect rect, PiiClass cls) {
    final color = RedactColors.forPii(cls);
    final rrect = RRect.fromRectAndRadius(rect, const Radius.circular(4));

    canvas.drawRRect(rrect, Paint()..color = RedactColors.redaction);
    canvas.drawRRect(
      rrect,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2
        ..color = color,
    );
    // Hatch stripe to read as "redacted", not "missing".
    final stripe = Paint()
      ..color = color.withValues(alpha: 0.35)
      ..strokeWidth = 1.5;
    canvas.save();
    canvas.clipRRect(rrect);
    for (double x = rect.left - rect.height;
        x < rect.right;
        x += 12) {
      canvas.drawLine(Offset(x, rect.bottom), Offset(x + rect.height, rect.top),
          stripe);
    }
    canvas.restore();

    _drawChip(canvas, rect, cls.label, color);
  }

  void _drawTextBox(Canvas canvas, Rect rect, RegionView region) {
    final bool read = region.isRead;
    final color = read ? RedactColors.textBox : RedactColors.accent;
    canvas.drawRRect(
      RRect.fromRectAndRadius(rect, const Radius.circular(4)),
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = read ? 1.6 : 1.0
        ..color = color.withValues(alpha: read ? 0.9 : 0.55),
    );
    final text = region.text;
    if (read && text != null && text.isNotEmpty) {
      _drawChip(canvas, rect, text, color);
    }
  }

  void _drawChip(Canvas canvas, Rect rect, String label, Color color) {
    final tp = TextPainter(
      text: TextSpan(
        text: ' $label ',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 11,
          fontWeight: FontWeight.w700,
        ),
      ),
      textDirection: TextDirection.ltr,
      ellipsis: '…',
    )..layout(maxWidth: 220);

    const double chipH = 16;
    final double chipW = tp.width + 2;
    double chipTop = rect.top - chipH - 1;
    if (chipTop < 0) chipTop = rect.bottom + 1;
    final chipRect = Rect.fromLTWH(rect.left, chipTop, chipW, chipH);
    canvas.drawRRect(
      RRect.fromRectAndRadius(chipRect, const Radius.circular(3)),
      Paint()..color = color.withValues(alpha: 0.85),
    );
    tp.paint(canvas, Offset(rect.left + 1, chipTop + (chipH - tp.height) / 2));
  }

  @override
  bool shouldRepaint(covariant RedactionOverlay old) {
    // Repaint only when the snapshot actually changed (Tier B: don't rebuild
    // the painter for identical frames).
    return !identical(old.regions, regions) ||
        old.frameWidth != frameWidth ||
        old.frameHeight != frameHeight;
  }
}

/// Optional blur layer used when [kRedactionStyle] == [RedactionStyle.blur]:
/// a BackdropFilter clipped to each redacted rect (heavier than bars; keep
/// off unless measured free on-device).
class RedactionBlurLayer extends StatelessWidget {
  const RedactionBlurLayer({
    super.key,
    required this.regions,
    required this.frameWidth,
    required this.frameHeight,
  });

  final List<RegionView> regions;
  final int frameWidth;
  final int frameHeight;

  @override
  Widget build(BuildContext context) {
    if (frameWidth <= 0 || frameHeight <= 0) return const SizedBox.shrink();
    return LayoutBuilder(
      builder: (context, constraints) {
        final mapper = FrameMapper(
          frameWidth: frameWidth.toDouble(),
          frameHeight: frameHeight.toDouble(),
          widgetSize: Size(constraints.maxWidth, constraints.maxHeight),
        );
        return Stack(
          children: [
            for (final region in regions)
              if (region.isRedacted)
                Positioned.fromRect(
                  rect: mapper.mapRect(region.bbox).inflate(2),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: BackdropFilter(
                      filter: ui.ImageFilter.blur(sigmaX: 12, sigmaY: 12),
                      child: const ColoredBox(color: Colors.transparent),
                    ),
                  ),
                ),
          ],
        );
      },
    );
  }
}
