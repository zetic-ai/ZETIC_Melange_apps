import 'dart:typed_data';

import '../models/geometry.dart';
import 'frame.dart';

/// PAD (MiniFASNet-V2) model input side length.
const int kPadInputSize = 80;

/// The PAD 2.7x face-margin crop, ported **verbatim** from the MiniVision
/// `CropImage` recipe in SPEC_STUB.md. This geometry is load-bearing: a tight or
/// wrong-scale crop collapses the model to a constant output (validated).
///
///   scale = min((H-1)/bh, (W-1)/bw, margin)
///   nw,nh = bw*scale, bh*scale ; centered on the bbox center
///   then clip-SHIFT (not clamp) so the box stays fully inside the frame.
///
/// [w],[h] are the upright frame dimensions; [box] is the face bbox in the same
/// upright space.
CropBox computeCropBox(int w, int h, FaceBox box, {double margin = 2.7}) {
  final double bw = box.w;
  final double bh = box.h;

  double scale = margin;
  final double sh = (h - 1) / bh;
  final double sw = (w - 1) / bw;
  if (sh < scale) scale = sh;
  if (sw < scale) scale = sw;

  final double nw = bw * scale;
  final double nh = bh * scale;
  final double cx = box.cx;
  final double cy = box.cy;

  double x1 = cx - nw / 2;
  double y1 = cy - nh / 2;
  double x2 = cx + nw / 2;
  double y2 = cy + nh / 2;

  // Clip-SHIFT: move the whole box in-bounds, preserving its size, rather than
  // clamping a single edge (which would distort the aspect ratio).
  if (x1 < 0) {
    x2 -= x1;
    x1 = 0;
  }
  if (y1 < 0) {
    y2 -= y1;
    y1 = 0;
  }
  if (x2 > w - 1) {
    x1 -= (x2 - (w - 1));
    x2 = (w - 1).toDouble();
  }
  if (y2 > h - 1) {
    y1 -= (y2 - (h - 1));
    y2 = (h - 1).toDouble();
  }
  return CropBox(x1, y1, x2, y2);
}

/// Builds the PAD input tensor: crop [box]'s 2.7x margin region from [img],
/// resize (bilinear) to 80x80, keep **BGR** channel order and **[0,255]** range
/// (⚠️ do NOT divide by 255 — that saturates the ONNX to a dead constant), and
/// lay out as NCHW (channel 0=B, 1=G, 2=R).
///
/// Pass [out] to reuse a pre-allocated buffer across frames (Tier B).
Float32List buildPadInput(BgrImage img, CropBox crop, {Float32List? out}) {
  const int size = kPadInputSize;
  const int area = size * size;
  final Float32List input = out ?? Float32List(3 * area);

  // Integer inclusive crop rectangle (matches Python frame[iy1:iy2+1, ix1:ix2+1]).
  final int ix1 = crop.x1.floor();
  final int iy1 = crop.y1.floor();
  final int ix2 = crop.x2.floor();
  final int iy2 = crop.y2.floor();
  final int cropW = ix2 - ix1 + 1;
  final int cropH = iy2 - iy1 + 1;

  for (var dy = 0; dy < size; dy++) {
    // Half-pixel-center bilinear mapping (align_corners=false).
    final double srcY = iy1 + (dy + 0.5) * cropH / size - 0.5;
    final int rowB = dy * size;
    for (var dx = 0; dx < size; dx++) {
      final double srcX = ix1 + (dx + 0.5) * cropW / size - 0.5;
      final Bgr px = bilinearBgr(img, srcX, srcY);
      final int idx = rowB + dx;
      input[idx] = px.b.toDouble(); // channel 0 = B
      input[area + idx] = px.g.toDouble(); // channel 1 = G
      input[2 * area + idx] = px.r.toDouble(); // channel 2 = R
    }
  }
  return input;
}

/// A resolved BGR triple.
class Bgr {
  const Bgr(this.b, this.g, this.r);
  final int b;
  final int g;
  final int r;
}

/// Bilinear sample of [img] at fractional absolute coords, clamped to bounds.
/// Returns rounded integer B,G,R in [0,255].
Bgr bilinearBgr(BgrImage img, double x, double y) {
  final int x0 = x.floor();
  final int y0 = y.floor();
  final double fx = x - x0;
  final double fy = y - y0;
  final int x1 = x0 + 1;
  final int y1 = y0 + 1;

  final double w00 = (1 - fx) * (1 - fy);
  final double w10 = fx * (1 - fy);
  final double w01 = (1 - fx) * fy;
  final double w11 = fx * fy;

  final double b = img.blueAt(x0, y0) * w00 +
      img.blueAt(x1, y0) * w10 +
      img.blueAt(x0, y1) * w01 +
      img.blueAt(x1, y1) * w11;
  final double g = img.greenAt(x0, y0) * w00 +
      img.greenAt(x1, y0) * w10 +
      img.greenAt(x0, y1) * w01 +
      img.greenAt(x1, y1) * w11;
  final double r = img.redAt(x0, y0) * w00 +
      img.redAt(x1, y0) * w10 +
      img.redAt(x0, y1) * w01 +
      img.redAt(x1, y1) * w11;

  return Bgr(
    b.round().clamp(0, 255),
    g.round().clamp(0, 255),
    r.round().clamp(0, 255),
  );
}
