import 'dart:math' as math;
import 'dart:typed_data';

import '../config.dart';
import 'quad_deskew.dart' show BgrCrop;

/// Fills [out] (pre-allocated, length 3·48·320) with the recognizer input for
/// one deskewed crop, and returns the resized content width.
///
/// SPEC-exact: aspect-preserving bilinear resize to height 48,
/// width = min(round(48·w/h), 320) — PAD, NEVER STRETCH — then normalize
/// (px/255 − 0.5)/0.5 into [−1,1], NCHW, BGR channel order. The right pad
/// (columns >= resizedW) stays 0.0 in NORMALIZED tensor space (GATE-2 ruling
/// #1: PaddleOCR pads the normalized tensor with zeros, NOT with black
/// pixels, which would be −1.0).
int recognizerPreprocess(BgrCrop crop, Float32List out) {
  const outH = kRecHeight;
  const outW = kRecWidth;
  const outArea = outH * outW;
  assert(out.length == 3 * outArea, 'recognizer input buffer must be 3*48*320');
  out.fillRange(0, out.length, 0.0);

  final srcW = crop.width;
  final srcH = crop.height;
  final resizedW = math.min((outH * srcW / srcH).round(), outW).clamp(1, outW);

  final scaleX = srcW / resizedW;
  final scaleY = srcH / outH;
  final bytes = crop.bgr;
  const inv = 1.0 / 255.0;

  for (var oy = 0; oy < outH; oy++) {
    var sy = (oy + 0.5) * scaleY - 0.5;
    if (sy < 0) sy = 0;
    var y0 = sy.floor();
    if (y0 > srcH - 2) y0 = math.max(0, srcH - 2);
    final fy = (sy - y0).clamp(0.0, 1.0);
    final y1 = math.min(y0 + 1, srcH - 1);
    final rowBase = oy * outW;

    for (var ox = 0; ox < resizedW; ox++) {
      var sx = (ox + 0.5) * scaleX - 0.5;
      if (sx < 0) sx = 0;
      var x0 = sx.floor();
      if (x0 > srcW - 2) x0 = math.max(0, srcW - 2);
      final fx = (sx - x0).clamp(0.0, 1.0);
      final x1 = math.min(x0 + 1, srcW - 1);

      final i00 = (y0 * srcW + x0) * 3;
      final i01 = (y0 * srcW + x1) * 3;
      final i10 = (y1 * srcW + x0) * 3;
      final i11 = (y1 * srcW + x1) * 3;
      final w00 = (1 - fx) * (1 - fy);
      final w01 = fx * (1 - fy);
      final w10 = (1 - fx) * fy;
      final w11 = fx * fy;

      final p = rowBase + ox;
      for (var c = 0; c < 3; c++) {
        final v = bytes[i00 + c] * w00 +
            bytes[i01 + c] * w01 +
            bytes[i10 + c] * w10 +
            bytes[i11 + c] * w11;
        // (v/255 - 0.5) / 0.5 == v/255*2 - 1
        out[c * outArea + p] = v * inv * 2.0 - 1.0;
      }
    }
  }
  return resizedW;
}
