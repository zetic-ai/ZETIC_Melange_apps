import 'dart:typed_data';

/// Model input side length. **928, NOT 640.** A stray 640 anywhere (here, the
/// NCHW buffer, or the letterbox inverse) silently misplaces every box.
const int kInputSize = 928;

/// Number of float elements in one NCHW input tensor: 1 * 3 * 928 * 928.
const int kInputElements = 3 * kInputSize * kInputSize;

/// Letterbox pad color, normalized. 114/255 is the standard Ultralytics gray.
const double kPadValue = 114.0 / 255.0;

/// Geometry of a letterbox-resize from a source frame into a square [target].
///
/// Aspect ratio is preserved; the scaled image is centered with symmetric
/// padding. The forward map (used by the fused preprocess) and the inverse map
/// (used by the postprocess) are exact reverses of each other.
class LetterboxParams {
  LetterboxParams({
    required this.srcW,
    required this.srcH,
    required this.target,
    required this.scale,
    required this.scaledW,
    required this.scaledH,
    required this.padX,
    required this.padY,
  });

  final int srcW;
  final int srcH;
  final int target;
  final double scale;
  final int scaledW;
  final int scaledH;
  final double padX;
  final double padY;

  /// Inverse map: a model-space (letterboxed) X back to source-pixel X.
  double modelToSrcX(double mx) => (mx - padX) / scale;

  /// Inverse map: a model-space (letterboxed) Y back to source-pixel Y.
  double modelToSrcY(double my) => (my - padY) / scale;
}

/// Compute letterbox geometry for a [srcW] x [srcH] frame into [target]x[target].
LetterboxParams computeLetterbox(int srcW, int srcH, {int target = kInputSize}) {
  final double scale =
      (target / srcW) < (target / srcH) ? target / srcW : target / srcH;
  final int scaledW = (srcW * scale).round();
  final int scaledH = (srcH * scale).round();
  final double padX = (target - scaledW) / 2.0;
  final double padY = (target - scaledH) / 2.0;
  return LetterboxParams(
    srcW: srcW,
    srcH: srcH,
    target: target,
    scale: scale,
    scaledW: scaledW,
    scaledH: scaledH,
    padX: padX,
    padY: padY,
  );
}

/// Fused resize + normalize + NCHW reorder for a packed RGB source.
///
/// One pass over the [kInputSize]x[kInputSize] destination: each destination
/// pixel is inverse-mapped to a source pixel (nearest-neighbour), normalized to
/// 0..1, and written straight into the preallocated NCHW [out] buffer. Pad
/// region is filled with [kPadValue]. No intermediate buffers.
///
/// [rgb] is row-major, 3 bytes/pixel (R,G,B), [srcW]*[srcH]*3 long.
/// [out] must be length [kInputElements] and is reused across frames.
void letterboxRgbToNchw(
  Uint8List rgb,
  int srcW,
  int srcH,
  LetterboxParams p,
  Float32List out,
) {
  const int size = kInputSize;
  const int channelStride = size * size; // H*W per channel
  final double invScale = 1.0 / p.scale;
  for (int dy = 0; dy < size; dy++) {
    final double syf = (dy - p.padY) * invScale;
    final int sy = syf.toInt();
    final bool rowInside = syf >= 0 && sy < srcH;
    final int rowBase = dy * size;
    for (int dx = 0; dx < size; dx++) {
      final int rIdx = rowBase + dx;
      final int gIdx = channelStride + rIdx;
      final int bIdx = 2 * channelStride + rIdx;
      if (rowInside) {
        final double sxf = (dx - p.padX) * invScale;
        final int sx = sxf.toInt();
        if (sxf >= 0 && sx < srcW) {
          final int off = (sy * srcW + sx) * 3;
          out[rIdx] = rgb[off] / 255.0;
          out[gIdx] = rgb[off + 1] / 255.0;
          out[bIdx] = rgb[off + 2] / 255.0;
          continue;
        }
      }
      out[rIdx] = kPadValue;
      out[gIdx] = kPadValue;
      out[bIdx] = kPadValue;
    }
  }
}

/// Fused letterbox for an iOS BGRA8888 buffer (the cheapest usable iOS format).
///
/// Samples BGRA directly and swizzles to RGB inline, so resize + channel-swap +
/// normalize + NCHW reorder are a single pass — no intermediate RGB buffer.
/// [bytesPerRow] may exceed [srcW]*4 (row padding), so it is honoured.
void letterboxBgraToNchw(
  Uint8List bgra,
  int srcW,
  int srcH,
  int bytesPerRow,
  LetterboxParams p,
  Float32List out,
) {
  const int size = kInputSize;
  const int channelStride = size * size;
  final double invScale = 1.0 / p.scale;
  for (int dy = 0; dy < size; dy++) {
    final double syf = (dy - p.padY) * invScale;
    final int sy = syf.toInt();
    final bool rowInside = syf >= 0 && sy < srcH;
    final int rowBase = dy * size;
    for (int dx = 0; dx < size; dx++) {
      final int rIdx = rowBase + dx;
      final int gIdx = channelStride + rIdx;
      final int bIdx = 2 * channelStride + rIdx;
      if (rowInside) {
        final double sxf = (dx - p.padX) * invScale;
        final int sx = sxf.toInt();
        if (sxf >= 0 && sx < srcW) {
          final int off = sy * bytesPerRow + sx * 4; // B,G,R,A
          out[rIdx] = bgra[off + 2] / 255.0;
          out[gIdx] = bgra[off + 1] / 255.0;
          out[bIdx] = bgra[off] / 255.0;
          continue;
        }
      }
      out[rIdx] = kPadValue;
      out[gIdx] = kPadValue;
      out[bIdx] = kPadValue;
    }
  }
}

/// Convert an Android YUV420 (planar/semi-planar) frame to packed RGB.
///
/// Android's secondary path: there is no single-pass BGRA equivalent, so YUV is
/// converted once here, then [letterboxRgbToNchw] does the resize. [yPlane],
/// [uPlane], [vPlane] are the raw plane bytes; the *RowStride/*PixelStride
/// describe their layout (UV planes are typically half-resolution).
Uint8List yuv420ToRgb(
  Uint8List yPlane,
  Uint8List uPlane,
  Uint8List vPlane,
  int srcW,
  int srcH,
  int yRowStride,
  int uvRowStride,
  int uvPixelStride,
) {
  final Uint8List rgb = Uint8List(srcW * srcH * 3);
  for (int y = 0; y < srcH; y++) {
    final int yRow = y * yRowStride;
    final int uvRow = (y >> 1) * uvRowStride;
    int outIdx = y * srcW * 3;
    for (int x = 0; x < srcW; x++) {
      final int yp = yPlane[yRow + x] & 0xFF;
      final int uvCol = (x >> 1) * uvPixelStride;
      final int up = (uPlane[uvRow + uvCol] & 0xFF) - 128;
      final int vp = (vPlane[uvRow + uvCol] & 0xFF) - 128;
      // BT.601 integer approximation.
      int r = yp + ((91881 * vp) >> 16);
      int g = yp - ((22554 * up + 46802 * vp) >> 16);
      int b = yp + ((116130 * up) >> 16);
      rgb[outIdx++] = r < 0 ? 0 : (r > 255 ? 255 : r);
      rgb[outIdx++] = g < 0 ? 0 : (g > 255 ? 255 : g);
      rgb[outIdx++] = b < 0 ? 0 : (b > 255 ? 255 : b);
    }
  }
  return rgb;
}
