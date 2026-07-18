import 'dart:typed_data';

import '../models/geometry.dart';
import 'frame.dart';
import 'preprocessor.dart' show bilinearBgr;

/// SFace / ArcFace model input side length.
const int kFaceInputSize = 112;

/// The canonical ArcFace 5-point template for a 112x112 aligned crop, in
/// (x, y) order: left eye, right eye, nose tip, left mouth corner, right mouth
/// corner. These are the standard insightface reference points.
const List<Point2> kArcFaceTemplate = [
  Point2(38.2946, 51.6963),
  Point2(73.5318, 51.5014),
  Point2(56.0252, 71.7366),
  Point2(41.5493, 92.3655),
  Point2(70.7299, 92.2041),
];

/// A 2-D similarity transform (uniform scale + rotation + translation, no
/// reflection): `x' = a*x - b*y + tx`, `y' = b*x + a*y + ty`.
class SimilarityTransform {
  const SimilarityTransform(this.a, this.b, this.tx, this.ty);

  final double a;
  final double b;
  final double tx;
  final double ty;

  Point2 apply(Point2 p) =>
      Point2(a * p.x - b * p.y + tx, b * p.x + a * p.y + ty);
}

/// Estimates the least-squares similarity transform mapping [from] onto [to]
/// (both must have the same length >= 2). Solves for [a, b, tx, ty] via the
/// 4x4 normal equations — the standard closed-form similarity fit (equivalent
/// to Umeyama without the reflection case, which face landmarks never need).
///
/// Each correspondence contributes two rows:
///   [ x, -y, 1, 0 ] . θ = X
///   [ y,  x, 0, 1 ] . θ = Y
SimilarityTransform estimateSimilarity(List<Point2> from, List<Point2> to) {
  assert(from.length == to.length && from.length >= 2);
  // Accumulate A^T A (4x4) and A^T b (4).
  final ata = List<double>.filled(16, 0);
  final atb = List<double>.filled(4, 0);

  void addRow(List<double> row, double rhs) {
    for (var i = 0; i < 4; i++) {
      atb[i] += row[i] * rhs;
      for (var j = 0; j < 4; j++) {
        ata[i * 4 + j] += row[i] * row[j];
      }
    }
  }

  for (var k = 0; k < from.length; k++) {
    final double x = from[k].x;
    final double y = from[k].y;
    addRow([x, -y, 1, 0], to[k].x);
    addRow([y, x, 0, 1], to[k].y);
  }

  final theta = _solve4x4(ata, atb);
  return SimilarityTransform(theta[0], theta[1], theta[2], theta[3]);
}

/// Solves a 4x4 linear system `M x = b` by Gaussian elimination with partial
/// pivoting. [m] is row-major length 16, [b] length 4.
List<double> _solve4x4(List<double> m, List<double> b) {
  // Work on copies.
  final a = List<double>.of(m);
  final rhs = List<double>.of(b);
  const n = 4;

  for (var col = 0; col < n; col++) {
    // Partial pivot.
    var pivot = col;
    var best = a[col * n + col].abs();
    for (var r = col + 1; r < n; r++) {
      final v = a[r * n + col].abs();
      if (v > best) {
        best = v;
        pivot = r;
      }
    }
    if (pivot != col) {
      for (var c = 0; c < n; c++) {
        final t = a[col * n + c];
        a[col * n + c] = a[pivot * n + c];
        a[pivot * n + c] = t;
      }
      final t = rhs[col];
      rhs[col] = rhs[pivot];
      rhs[pivot] = t;
    }
    final diag = a[col * n + col];
    for (var r = 0; r < n; r++) {
      if (r == col) continue;
      final factor = a[r * n + col] / diag;
      if (factor == 0) continue;
      for (var c = col; c < n; c++) {
        a[r * n + c] -= factor * a[col * n + c];
      }
      rhs[r] -= factor * rhs[col];
    }
  }

  return [
    for (var i = 0; i < n; i++) rhs[i] / a[i * n + i],
  ];
}

/// Builds the SFace input tensor: warp [img] to a 112x112 ArcFace-aligned crop
/// using the 5 landmarks, keep **BGR** channel order and **[0,255]** range (the
/// (x-127.5)/128 normalization is baked into the ONNX graph — do NOT normalize
/// in Dart), and lay out as NCHW (channel 0=B, 1=G, 2=R).
///
/// The transform maps template coordinates -> source coordinates, so each output
/// pixel samples directly from the source with no matrix inversion.
///
/// Pass [out] to reuse a pre-allocated buffer across frames (Tier B).
Float32List buildFaceInput(BgrImage img, Landmarks5 lm, {Float32List? out}) {
  const int size = kFaceInputSize;
  const int area = size * size;
  final Float32List input = out ?? Float32List(3 * area);

  // template -> source, so sampling output(u,v) reads source T(u,v).
  final SimilarityTransform t =
      estimateSimilarity(kArcFaceTemplate, lm.ordered);

  for (var v = 0; v < size; v++) {
    final int rowB = v * size;
    for (var u = 0; u < size; u++) {
      final Point2 src = t.apply(Point2(u.toDouble(), v.toDouble()));
      final px = bilinearBgr(img, src.x, src.y);
      final int idx = rowB + u;
      input[idx] = px.b.toDouble();
      input[area + idx] = px.g.toDouble();
      input[2 * area + idx] = px.r.toDouble();
    }
  }
  return input;
}
