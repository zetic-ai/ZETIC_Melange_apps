import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Offset;

import '../config.dart';
import '../models/text_region.dart';
import 'detector_preprocessor.dart' show BgrFrame;

/// An upright BGR crop produced by perspective-deskewing a text quad.
class BgrCrop {
  BgrCrop(this.width, this.height, this.bgr)
      : assert(bgr.length == width * height * 3, 'crop buffer size mismatch');

  final int width;
  final int height;
  final Uint8List bgr;
}

/// Solves the 3×3 homography H (h33 = 1) that maps DESTINATION rectangle
/// coordinates (0,0)-(w,0)-(w,h)-(0,h) onto the source [quad] corners
/// tl-tr-br-bl, via an 8×8 linear system (Gaussian elimination, no deps).
///
/// Returns row-major [h11..h32] (8 values).
Float64List computeHomography(Quad quad, double w, double h) {
  final dst = [
    const Offset(0, 0),
    Offset(w, 0),
    Offset(w, h),
    Offset(0, h),
  ];
  final src = quad.points;

  // For each correspondence (x,y) -> (X,Y):
  //   h11 x + h12 y + h13 - h31 x X - h32 y X = X
  //   h21 x + h22 y + h23 - h31 x Y - h32 y Y = Y
  final a = List.generate(8, (_) => Float64List(8));
  final b = Float64List(8);
  for (var i = 0; i < 4; i++) {
    final x = dst[i].dx, y = dst[i].dy;
    final xx = src[i].dx, yy = src[i].dy;
    a[2 * i][0] = x;
    a[2 * i][1] = y;
    a[2 * i][2] = 1;
    a[2 * i][6] = -x * xx;
    a[2 * i][7] = -y * xx;
    b[2 * i] = xx;
    a[2 * i + 1][3] = x;
    a[2 * i + 1][4] = y;
    a[2 * i + 1][5] = 1;
    a[2 * i + 1][6] = -x * yy;
    a[2 * i + 1][7] = -y * yy;
    b[2 * i + 1] = yy;
  }
  return _solve8(a, b);
}

/// Applies homography [hm] ([h11..h32], h33=1) to a destination point.
Offset applyHomography(Float64List hm, double x, double y) {
  final denom = hm[6] * x + hm[7] * y + 1;
  return Offset(
    (hm[0] * x + hm[1] * y + hm[2]) / denom,
    (hm[3] * x + hm[4] * y + hm[5]) / denom,
  );
}

/// Perspective-warps [quad] (upright-frame space) out of [frame] into an
/// upright axis-aligned BGR crop, bilinear-sampled. Destination size follows
/// PaddleOCR's get_rotate_crop_image: width = max of the two horizontal edge
/// lengths, height = max of the two vertical edge lengths.
///
/// PaddleOCR parity (GATE-2 ruling #2): crops with h >= [kVerticalCropRatio]·w
/// are rotated 90° counter-clockwise before recognition (vertical signage).
BgrCrop deskewQuad(BgrFrame frame, Quad quad) {
  double dist(Offset a, Offset b) => (a - b).distance;

  final w = math
      .max(dist(quad.tl, quad.tr), dist(quad.bl, quad.br))
      .round()
      .clamp(4, 4096);
  final h = math
      .max(dist(quad.tl, quad.bl), dist(quad.tr, quad.br))
      .round()
      .clamp(4, 4096);

  final hm = computeHomography(quad, w.toDouble(), h.toDouble());
  final out = Uint8List(w * h * 3);
  final srcW = frame.width;
  final srcH = frame.height;
  final bytes = frame.bgr;

  var o = 0;
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      // Map the dest pixel center; sample source pixel centers bilinearly.
      final s = applyHomography(hm, x + 0.5, y + 0.5);
      final sx = (s.dx - 0.5).clamp(0.0, srcW - 1.0);
      final sy = (s.dy - 0.5).clamp(0.0, srcH - 1.0);
      var x0 = sx.floor();
      var y0 = sy.floor();
      if (x0 > srcW - 2) x0 = math.max(0, srcW - 2);
      if (y0 > srcH - 2) y0 = math.max(0, srcH - 2);
      final fx = sx - x0;
      final fy = sy - y0;
      final x1 = math.min(x0 + 1, srcW - 1);
      final y1 = math.min(y0 + 1, srcH - 1);

      final i00 = (y0 * srcW + x0) * 3;
      final i01 = (y0 * srcW + x1) * 3;
      final i10 = (y1 * srcW + x0) * 3;
      final i11 = (y1 * srcW + x1) * 3;
      final w00 = (1 - fx) * (1 - fy);
      final w01 = fx * (1 - fy);
      final w10 = (1 - fx) * fy;
      final w11 = fx * fy;

      for (var c = 0; c < 3; c++) {
        out[o++] = (bytes[i00 + c] * w00 +
                bytes[i01 + c] * w01 +
                bytes[i10 + c] * w10 +
                bytes[i11 + c] * w11)
            .round()
            .clamp(0, 255);
      }
    }
  }

  final crop = BgrCrop(w, h, out);
  if (h >= kVerticalCropRatio * w) return rotateCrop90Ccw(crop);
  return crop;
}

/// Rotates a BGR crop 90° counter-clockwise (np.rot90 parity).
BgrCrop rotateCrop90Ccw(BgrCrop crop) {
  final w = crop.width, h = crop.height;
  // np.rot90: out[x][y] has shape (w, h) — new width = h, new height = w.
  final out = Uint8List(w * h * 3);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      // Source (x, y) -> dest (y, w-1-x) in the (h wide, w tall) output.
      final src = (y * w + x) * 3;
      final dst = ((w - 1 - x) * h + y) * 3;
      out[dst] = crop.bgr[src];
      out[dst + 1] = crop.bgr[src + 1];
      out[dst + 2] = crop.bgr[src + 2];
    }
  }
  return BgrCrop(h, w, out);
}

/// Gaussian elimination with partial pivoting for the 8×8 homography system.
Float64List _solve8(List<Float64List> a, Float64List b) {
  const n = 8;
  for (var col = 0; col < n; col++) {
    // Pivot.
    var pivot = col;
    for (var r = col + 1; r < n; r++) {
      if (a[r][col].abs() > a[pivot][col].abs()) pivot = r;
    }
    if (a[pivot][col].abs() < 1e-12) {
      throw StateError('Degenerate quad: homography system is singular');
    }
    if (pivot != col) {
      final tmp = a[pivot];
      a[pivot] = a[col];
      a[col] = tmp;
      final tb = b[pivot];
      b[pivot] = b[col];
      b[col] = tb;
    }
    // Eliminate.
    for (var r = col + 1; r < n; r++) {
      final f = a[r][col] / a[col][col];
      if (f == 0) continue;
      for (var c = col; c < n; c++) {
        a[r][c] -= f * a[col][c];
      }
      b[r] -= f * b[col];
    }
  }
  // Back-substitute.
  final x = Float64List(n);
  for (var r = n - 1; r >= 0; r--) {
    var sum = b[r];
    for (var c = r + 1; c < n; c++) {
      sum -= a[r][c] * x[c];
    }
    x[r] = sum / a[r][r];
  }
  return x;
}
