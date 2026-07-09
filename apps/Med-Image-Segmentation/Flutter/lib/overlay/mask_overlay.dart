import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

/// Rasterize a [w]x[h] binary mask (0/1) into a tinted RGBA [ui.Image] — LV
/// pixels get [color] at [color.alpha], the rest transparent.
///
/// [rgbaScratch] (length w*h*4) is filled and handed to the engine; the caller
/// awaits the returned future before reusing it, so a single buffer can be shared
/// across frames to avoid a per-frame allocation. The returned [ui.Image] holds
/// native memory and MUST be disposed once off-screen.
Future<ui.Image> buildMaskImage(
  Uint8List mask,
  int w,
  int h,
  Uint8List rgbaScratch, {
  Color color = const Color(0xB0FF3B30),
}) {
  final r = (color.r * 255).round();
  final g = (color.g * 255).round();
  final b = (color.b * 255).round();
  final a = (color.a * 255).round();
  for (var i = 0; i < mask.length; i++) {
    final o = i * 4;
    if (mask[i] != 0) {
      rgbaScratch[o] = r;
      rgbaScratch[o + 1] = g;
      rgbaScratch[o + 2] = b;
      rgbaScratch[o + 3] = a;
    } else {
      rgbaScratch[o + 3] = 0; // clear alpha from the previous frame's mask
    }
  }
  final c = Completer<ui.Image>();
  ui.decodeImageFromPixels(
    rgbaScratch,
    w,
    h,
    ui.PixelFormat.rgba8888,
    c.complete,
  );
  return c.future;
}

/// Draws the mask image scaled to fill the paint box (the box already matches the
/// square display frame, so a simple src->dst rect keeps the mask aligned).
class MaskPainter extends CustomPainter {
  MaskPainter(this.mask);
  final ui.Image? mask;

  @override
  void paint(Canvas canvas, Size size) {
    final m = mask;
    if (m == null) return;
    final src = Rect.fromLTWH(0, 0, m.width.toDouble(), m.height.toDouble());
    final dst = Offset.zero & size;
    canvas.drawImageRect(
      m,
      src,
      dst,
      Paint()..filterQuality = FilterQuality.low,
    );
  }

  @override
  bool shouldRepaint(covariant MaskPainter old) => old.mask != mask;
}
