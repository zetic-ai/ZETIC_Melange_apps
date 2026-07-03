import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui';

import 'frame_data.dart';

/// Recognizer input is a fixed [1, 3, 48, 320] (static shape — variable width
/// is a static-shape violation on Melange).
const int kRecHeight = 48;
const int kRecWidth = 320;
const int kRecTensorLength = 3 * kRecHeight * kRecWidth;

/// Warps one detected text quad upright and builds the recognizer input
/// tensor, PaddleOCR-rec style:
///
/// 1. The quad (tl, tr, br, bl in upright source pixels) is deskewed so text
///    reads left-to-right: output row 0 runs tl -> tr.
/// 2. Aspect-preserving resize to height 48; target width = round(48 * w/h),
///    capped at 320 (wider crops are downscaled to exactly 320 — pad, never
///    stretch).
/// 3. Pixels are normalized (pixel/255 - 0.5)/0.5 -> [-1, 1] on BGR.
/// 4. Columns beyond the content width are zero in NORMALIZED space (0.0),
///    matching PaddleOCR's `padding_im = np.zeros(...)` semantics.
///
/// The crop samples the ORIGINAL full-resolution frame (not the letterboxed
/// 640 tensor), via bilinear interpolation of the quad corners — exact for
/// the rotated rectangles the DB decoder emits.
///
/// [out] may be a reused [kRecTensorLength] buffer.
Float32List preprocessRecognizerCrop(UprightFrame src, List<Offset> quad,
    {Float32List? out}) {
  assert(quad.length == 4);
  final Float32List input = out ?? Float32List(kRecTensorLength);
  assert(input.length == kRecTensorLength);

  final Offset tl = quad[0], tr = quad[1], br = quad[2], bl = quad[3];
  final double srcW =
      (((tr - tl).distance) + ((br - bl).distance)) / 2.0;
  final double srcH =
      (((bl - tl).distance) + ((br - tr).distance)) / 2.0;
  if (srcW < 1 || srcH < 1) {
    input.fillRange(0, kRecTensorLength, 0.0);
    return input;
  }

  int outW = (kRecHeight * srcW / srcH).round();
  if (outW > kRecWidth) outW = kRecWidth;
  if (outW < 1) outW = 1;

  // Zero the whole tensor first: the right pad must be 0.0 (normalized space)
  // and the buffer may be reused across crops.
  input.fillRange(0, kRecTensorLength, 0.0);

  const int area = kRecHeight * kRecWidth;
  const double inv = 1.0 / 127.5; // (px/255 - 0.5)/0.5 == px/127.5 - 1

  for (var oy = 0; oy < kRecHeight; oy++) {
    final double v = (oy + 0.5) / kRecHeight;
    final int rowBase = oy * kRecWidth;
    // Row endpoints: left edge tl->bl, right edge tr->br.
    final double lx = tl.dx + (bl.dx - tl.dx) * v;
    final double ly = tl.dy + (bl.dy - tl.dy) * v;
    final double rx = tr.dx + (br.dx - tr.dx) * v;
    final double ry = tr.dy + (br.dy - tr.dy) * v;
    for (var ox = 0; ox < outW; ox++) {
      final double u = (ox + 0.5) / outW;
      final int sx = (lx + (rx - lx) * u).round();
      final int sy = (ly + (ry - ly) * u).round();

      final int bgr = src.sampleBgrPacked(sx, sy);
      final int b = (bgr >> 16) & 0xff;
      final int g = (bgr >> 8) & 0xff;
      final int r = bgr & 0xff;

      final int p = rowBase + ox;
      input[p] = b * inv - 1.0; // channel 0 = B
      input[area + p] = g * inv - 1.0; // channel 1 = G
      input[2 * area + p] = r * inv - 1.0; // channel 2 = R
    }
  }

  return input;
}

/// Content width (pre-padding) the recognizer preprocessor will produce for a
/// quad — exposed for tests and the scheduler.
int recognizerContentWidth(List<Offset> quad) {
  final Offset tl = quad[0], tr = quad[1], br = quad[2], bl = quad[3];
  final double srcW = (((tr - tl).distance) + ((br - bl).distance)) / 2.0;
  final double srcH = (((bl - tl).distance) + ((br - tr).distance)) / 2.0;
  if (srcW < 1 || srcH < 1) return 1;
  return math.max(1, math.min(kRecWidth, (kRecHeight * srcW / srcH).round()));
}
