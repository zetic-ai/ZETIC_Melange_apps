import 'dart:typed_data';
import 'dart:ui' show Offset;

import '../config.dart';
import 'frame_data.dart';

/// An upright interleaved BGR frame (3 bytes per pixel, row-major).
///
/// Built once per detection pass and retained by the pipeline isolate: the
/// detector letterbox samples from it, and every per-crop deskew warps from
/// it (so staggered recognition frames need no camera frame at all).
class BgrFrame {
  BgrFrame(this.width, this.height, this.bgr)
      : assert(bgr.length == width * height * 3, 'BGR buffer size mismatch');

  final int width;
  final int height;

  /// Interleaved B,G,R bytes — PaddleOCR keeps cv2 BGR order end-to-end;
  /// nothing in this pipeline ever swaps to RGB.
  final Uint8List bgr;
}

/// Converts (and uprights, per [FrameData.rotationDegrees]) a camera frame to
/// interleaved BGR. iOS BGRA drops alpha; Android YUV420 converts via BT.601.
///
/// Tier-B: the BGRA path avoids the per-pixel [uprightToRaw] call (a record
/// allocation + switch per pixel) by hoisting the rotation into a per-row
/// start index and a constant per-pixel stride through the raw buffer.
BgrFrame convertToUprightBgr(FrameData frame) {
  final rawW = frame.width;
  final rawH = frame.height;
  final rot = ((frame.rotationDegrees % 360) + 360) % 360;
  final swap = rot == 90 || rot == 270;
  final w = swap ? rawH : rawW;
  final h = swap ? rawW : rawH;
  final out = Uint8List(w * h * 3);

  if (frame.format == FrameFormat.bgra8888) {
    final bytes = frame.bgra!;
    final stride = frame.bgraRowStride;
    // Raw-buffer index of upright (0, uy) and the index step per +1 ux,
    // derived from the same mapping as uprightToRaw (verified round-trip in
    // the orientation tests).
    for (var uy = 0; uy < h; uy++) {
      int idx;
      int step;
      switch (rot) {
        case 90: // (rx, ry) = (uy, rawH-1-ux)
          idx = (rawH - 1) * stride + uy * 4;
          step = -stride;
        case 180: // (rx, ry) = (rawW-1-ux, rawH-1-uy)
          idx = (rawH - 1 - uy) * stride + (rawW - 1) * 4;
          step = -4;
        case 270: // (rx, ry) = (rawW-1-uy, ux)
          idx = (rawW - 1 - uy) * 4;
          step = stride;
        default: // 0: (rx, ry) = (ux, uy)
          idx = uy * stride;
          step = 4;
      }
      var o = uy * w * 3;
      for (var ux = 0; ux < w; ux++) {
        out[o] = bytes[idx];
        out[o + 1] = bytes[idx + 1];
        out[o + 2] = bytes[idx + 2];
        o += 3;
        idx += step;
      }
    }
    return BgrFrame(w, h, out);
  }

  // YUV420 (Android): general per-pixel path — the chroma index depends on
  // (rx>>1, ry>>1) so the stride trick does not apply cleanly.
  final yPlane = frame.yPlane!;
  final uPlane = frame.uPlane!;
  final vPlane = frame.vPlane!;
  var o = 0;
  for (var uy = 0; uy < h; uy++) {
    for (var ux = 0; ux < w; ux++) {
      final (rx, ry) = uprightToRaw(ux, uy, rawW, rawH, rot);
      final yIndex = ry * frame.yRowStride + rx;
      final uvIndex =
          (ry >> 1) * frame.uvRowStride + (rx >> 1) * frame.uvPixelStride;
      final yv = yPlane[yIndex];
      final uv = uPlane[uvIndex] - 128;
      final vv = vPlane[uvIndex] - 128;
      // BT.601 YUV -> BGR.
      out[o++] = (yv + 1.732446 * uv).round().clamp(0, 255);
      out[o++] = (yv - 0.337633 * uv - 0.698001 * vv).round().clamp(0, 255);
      out[o++] = (yv + 1.370705 * vv).round().clamp(0, 255);
    }
  }
  return BgrFrame(w, h, out);
}

/// Letterbox geometry: `model = frame * scale + pad` (forward) and its exact
/// inverse `frame = (model - pad) / scale`. The inverse deliberately reverses
/// the forward steps in reverse order (subtract pad, then divide by scale).
class LetterboxGeometry {
  const LetterboxGeometry({
    required this.scale,
    required this.padX,
    required this.padY,
    required this.srcWidth,
    required this.srcHeight,
  });

  final double scale;
  final int padX;
  final int padY;
  final int srcWidth;
  final int srcHeight;

  /// Frame space -> 736×736 letterboxed model space.
  Offset toModel(Offset framePoint) => Offset(
        framePoint.dx * scale + padX,
        framePoint.dy * scale + padY,
      );

  /// 736×736 letterboxed model space -> frame space (exact inverse).
  Offset toFrame(Offset modelPoint) => Offset(
        (modelPoint.dx - padX) / scale,
        (modelPoint.dy - padY) / scale,
      );
}

/// Computes the letterbox geometry for fitting `srcW`×`srcH` into
/// [kDetInputSize]² preserving aspect, padding centered.
LetterboxGeometry computeLetterboxGeometry(int srcW, int srcH) {
  const size = kDetInputSize;
  final scale = (size / srcW) < (size / srcH) ? size / srcW : size / srcH;
  final newW = (srcW * scale).round();
  final newH = (srcH * scale).round();
  return LetterboxGeometry(
    scale: scale,
    padX: (size - newW) ~/ 2,
    padY: (size - newH) ~/ 2,
    srcWidth: srcW,
    srcHeight: srcH,
  );
}

/// Fills [out] (pre-allocated, length 3·736·736) with the letterboxed,
/// BGR-ordered, ImageNet-normalized NCHW detector input in a single fused
/// pass (bilinear resample + /255 + mean/std + planar reorder).
///
/// Padding is 0.0 in normalized tensor space (np.zeros-style padding,
/// consistent with the recognizer's SPEC-ruled pad value).
LetterboxGeometry letterboxDetectorInput(BgrFrame src, Float32List out) {
  const size = kDetInputSize;
  const area = size * size;
  assert(out.length == 3 * area, 'detector input buffer must be 3*736*736');
  out.fillRange(0, out.length, 0.0);

  final geo = computeLetterboxGeometry(src.width, src.height);
  final newW = (src.width * geo.scale).round();
  final newH = (src.height * geo.scale).round();
  final invScale = 1 / geo.scale;
  final bytes = src.bgr;
  final srcW = src.width;
  final srcH = src.height;

  // Hoisted per-channel affine: (v/255 - mean)/std == v*a + b.
  final a0 = 1 / (255.0 * kDetStd[0]), b0 = -kDetMean[0] / kDetStd[0];
  final a1 = 1 / (255.0 * kDetStd[1]), b1 = -kDetMean[1] / kDetStd[1];
  final a2 = 1 / (255.0 * kDetStd[2]), b2 = -kDetMean[2] / kDetStd[2];

  for (var oy = 0; oy < newH; oy++) {
    // Pixel-center bilinear mapping.
    var sy = (oy + 0.5) * invScale - 0.5;
    if (sy < 0) sy = 0;
    var y0 = sy.floor();
    if (y0 > srcH - 2) y0 = srcH - 2;
    if (y0 < 0) y0 = 0;
    final fy = (sy - y0).clamp(0.0, 1.0);
    final row0 = y0 * srcW * 3;
    final row1 = (y0 + 1 < srcH ? y0 + 1 : y0) * srcW * 3;
    final outRow = (geo.padY + oy) * size + geo.padX;

    for (var ox = 0; ox < newW; ox++) {
      var sx = (ox + 0.5) * invScale - 0.5;
      if (sx < 0) sx = 0;
      var x0 = sx.floor();
      if (x0 > srcW - 2) x0 = srcW - 2;
      if (x0 < 0) x0 = 0;
      final fx = (sx - x0).clamp(0.0, 1.0);
      final x1 = x0 + 1 < srcW ? x0 + 1 : x0;

      final i00 = row0 + x0 * 3;
      final i01 = row0 + x1 * 3;
      final i10 = row1 + x0 * 3;
      final i11 = row1 + x1 * 3;
      final w00 = (1 - fx) * (1 - fy);
      final w01 = fx * (1 - fy);
      final w10 = (1 - fx) * fy;
      final w11 = fx * fy;

      final b = bytes[i00] * w00 +
          bytes[i01] * w01 +
          bytes[i10] * w10 +
          bytes[i11] * w11;
      final g = bytes[i00 + 1] * w00 +
          bytes[i01 + 1] * w01 +
          bytes[i10 + 1] * w10 +
          bytes[i11 + 1] * w11;
      final r = bytes[i00 + 2] * w00 +
          bytes[i01 + 2] * w01 +
          bytes[i10 + 2] * w10 +
          bytes[i11 + 2] * w11;

      final p = outRow + ox;
      out[p] = b * a0 + b0; // channel 0 = B
      out[area + p] = g * a1 + b1; // channel 1 = G
      out[2 * area + p] = r * a2 + b2; // channel 2 = R
    }
  }
  return geo;
}
