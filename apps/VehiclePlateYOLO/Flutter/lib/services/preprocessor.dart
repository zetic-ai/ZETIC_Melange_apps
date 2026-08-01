import 'dart:typed_data';

import 'frame_data.dart';
import 'letterbox.dart';
import 'orientation.dart';

/// Result of preprocessing one frame: the packed model input plus the geometry
/// needed to undo the letterbox in post-processing.
class PreprocessResult {
  const PreprocessResult(this.input, this.params);

  /// Float32 [1,3,640,640], NCHW, RGB, normalized 0..1. Pre-allocated and
  /// reused across frames (see [Preprocessor.buffer]).
  final Float32List input;
  final LetterboxParams params;
}

/// Raw frame -> float32 [1,3,640,640] NCHW RGB in a single fused pass.
///
/// Fuses letterbox-resize + orientation + pixel-format->RGB + /255 normalize +
/// NCHW pack into one reverse-mapped loop over the 640x640 output (no
/// intermediate buffers, Tier B "fuse ... single pass"). The input Float32List
/// is allocated once and overwritten in place (Tier B "pre-allocate").
class Preprocessor {
  Preprocessor({this.target = 640})
    : buffer = Float32List(3 * target * target);

  final int target;

  /// Reused every frame; never reallocated.
  final Float32List buffer;

  /// Normalized pad value (gray) for letterbox borders. Spec: "pad 0.5".
  static const double _pad = 0.5;

  /// Reciprocal of 255 so the per-pixel normalize is a multiply, not a divide.
  static const double _inv255 = 1.0 / 255.0;

  PreprocessResult process(FrameData frame) {
    final orient = FrameOrientation(frame.rotationDegrees);
    final bufW = frame.width;
    final bufH = frame.height;
    final srcW = orient.uprightWidth(bufW, bufH);
    final srcH = orient.uprightHeight(bufW, bufH);
    final params = LetterboxParams.forImage(srcW, srcH, target);

    final plane = target * target; // per-channel stride in NCHW
    final gPlane = plane;
    final bPlane = 2 * plane;
    final out = buffer;
    final scale = params.scale;
    final padX = params.padX;
    final padY = params.padY;

    // Hoist the format branch (and BGRA's plane/stride) out of the per-pixel
    // loop. The BGRA path (iOS hot path) reads bytes inline, with no method
    // call or intermediate list per pixel.
    final isBgra = frame.format == FramePixelFormat.bgra8888;
    final p0 = frame.plane0;
    final bpr0 = frame.bytesPerRow0;

    for (var oy = 0; oy < target; oy++) {
      final srcYf = (oy - padY) / scale;
      final rowBase = oy * target;
      final insideY = srcYf >= 0 && srcYf < srcH;
      final uy = srcYf.toInt();
      for (var ox = 0; ox < target; ox++) {
        final idx = rowBase + ox;
        if (!insideY) {
          out[idx] = _pad;
          out[gPlane + idx] = _pad;
          out[bPlane + idx] = _pad;
          continue;
        }
        final srcXf = (ox - padX) / scale;
        if (srcXf < 0 || srcXf >= srcW) {
          out[idx] = _pad;
          out[gPlane + idx] = _pad;
          out[bPlane + idx] = _pad;
          continue;
        }
        final ux = srcXf.toInt();
        // upright pixel -> raw buffer pixel
        final bp = orient.uprightToBuffer(ux, uy, bufW, bufH);
        final bx = bp[0];
        final by = bp[1];
        if (isBgra) {
          final o = by * bpr0 + bx * 4;
          out[idx] = p0[o + 2] * _inv255; // R
          out[gPlane + idx] = p0[o + 1] * _inv255; // G
          out[bPlane + idx] = p0[o] * _inv255; // B
        } else {
          final rgb = _sampleYuv(frame, bx, by);
          out[idx] = rgb[0] * _inv255;
          out[gPlane + idx] = rgb[1] * _inv255;
          out[bPlane + idx] = rgb[2] * _inv255;
        }
      }
    }
    return PreprocessResult(out, params);
  }

  /// Reused fixed list to avoid per-pixel allocation on the YUV path.
  final Int32List _rgb = Int32List(3);

  /// YUV420 (Android) -> [r,g,b] 0..255 for one buffer pixel. BGRA is read
  /// inline in [process]; this path is the Android (Tier C) one.
  Int32List _sampleYuv(FrameData f, int bx, int by) {
    final yVal = f.plane0[by * f.bytesPerRow0 + bx];
    final cx = bx >> 1;
    final cy = by >> 1;
    final uIdx = cy * f.bytesPerRow1 + cx * f.pixelStride1;
    final vIdx = cy * f.bytesPerRow2 + cx * f.pixelStride2;
    final u = f.plane1![uIdx] - 128;
    final v = f.plane2![vIdx] - 128;
    // BT.601 full-range YUV -> RGB
    final r = (yVal + 1.402 * v).round();
    final g = (yVal - 0.344136 * u - 0.714136 * v).round();
    final b = (yVal + 1.772 * u).round();
    _rgb[0] = r < 0 ? 0 : (r > 255 ? 255 : r);
    _rgb[1] = g < 0 ? 0 : (g > 255 ? 255 : g);
    _rgb[2] = b < 0 ? 0 : (b > 255 ? 255 : b);
    return _rgb;
  }
}
