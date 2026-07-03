import 'dart:typed_data';
import 'dart:ui';

import 'frame_data.dart';

/// DBNet detector input is 640x640.
const int kDetInputSize = 640;

/// PaddleOCR det normalization: (pixel/255 - mean) / std, applied INDEX-WISE
/// to the BGR channels exactly as PaddleOCR does (cv2 convention) — channel 0
/// is B with mean 0.485, etc. Do NOT reorder to RGB (SPEC-binding).
const List<double> kDetMean = [0.485, 0.456, 0.406];
const List<double> kDetStd = [0.229, 0.224, 0.225];

/// Letterbox bookkeeping needed to invert model-space (640x640) coordinates
/// back to upright source-frame pixels.
class LetterboxGeometry {
  const LetterboxGeometry({
    required this.scale,
    required this.padX,
    required this.padY,
    required this.srcWidth,
    required this.srcHeight,
  });

  /// Source-pixels -> 640 scale factor applied during letterboxing.
  final double scale;
  final int padX;
  final int padY;
  final int srcWidth;
  final int srcHeight;

  /// Maps a point in 640x640 model space back to upright source-frame pixels
  /// (the exact inverse of the forward letterbox), clamped to the frame.
  Offset unmap(Offset p) {
    final double x = ((p.dx - padX) / scale).clamp(0.0, srcWidth.toDouble());
    final double y = ((p.dy - padY) / scale).clamp(0.0, srcHeight.toDouble());
    return Offset(x, y);
  }
}

/// Result of detector preprocessing: the NCHW tensor plus letterbox geometry.
class DetPreprocessResult {
  const DetPreprocessResult({required this.input, required this.geometry});

  /// Flattened float32 buffer, shape [1, 3, 640, 640], NCHW, BGR,
  /// ImageNet-normalized.
  final Float32List input;
  final LetterboxGeometry geometry;
}

/// Builds the letterboxed, BGR ImageNet-normalized, planar NCHW detector
/// input in a single fused pass over the 640x640 output grid (nearest-neighbor
/// sampling; resize + normalize + reorder in one loop, no intermediates).
///
/// Padding is normalized black — (0 - mean)/std per channel — so the padded
/// border reads as a dark, textless margin to the DB head.
///
/// [out] may be a reused 3*640*640 buffer to avoid per-frame allocation.
DetPreprocessResult preprocessDetectorFrame(UprightFrame src,
    {Float32List? out}) {
  const int size = kDetInputSize;
  const int area = size * size;
  final Float32List input = out ?? Float32List(3 * area);
  assert(input.length == 3 * area);

  // Per-channel affine: v = pixel * scale_c + bias_c.
  final chScale = List<double>.generate(3, (c) => 1.0 / (255.0 * kDetStd[c]));
  final chBias = List<double>.generate(3, (c) => -kDetMean[c] / kDetStd[c]);

  final int srcW = src.width;
  final int srcH = src.height;
  final double scale =
      (size / srcW) < (size / srcH) ? (size / srcW) : (size / srcH);
  final int newW = (srcW * scale).round();
  final int newH = (srcH * scale).round();
  final int padX = (size - newW) ~/ 2;
  final int padY = (size - newH) ~/ 2;

  // Fill everything with normalized black, then overwrite the content window.
  final double padB = chBias[0];
  final double padG = chBias[1];
  final double padR = chBias[2];
  input.fillRange(0, area, padB);
  input.fillRange(area, 2 * area, padG);
  input.fillRange(2 * area, 3 * area, padR);

  final double sB = chScale[0], bB = chBias[0];
  final double sG = chScale[1], bG = chBias[1];
  final double sR = chScale[2], bR = chBias[2];

  for (var oy = padY; oy < padY + newH; oy++) {
    if (oy < 0 || oy >= size) continue;
    final int uy = ((oy - padY) / scale).floor().clamp(0, srcH - 1);
    final int rowBase = oy * size;
    for (var ox = padX; ox < padX + newW; ox++) {
      if (ox < 0 || ox >= size) continue;
      final int ux = ((ox - padX) / scale).floor().clamp(0, srcW - 1);

      final int bgr = src.sampleBgrPacked(ux, uy);
      final int b = (bgr >> 16) & 0xff;
      final int g = (bgr >> 8) & 0xff;
      final int r = bgr & 0xff;

      final int p = rowBase + ox;
      input[p] = b * sB + bB; // channel 0 = B
      input[area + p] = g * sG + bG; // channel 1 = G
      input[2 * area + p] = r * sR + bR; // channel 2 = R
    }
  }

  return DetPreprocessResult(
    input: input,
    geometry: LetterboxGeometry(
      scale: scale,
      padX: padX,
      padY: padY,
      srcWidth: srcW,
      srcHeight: srcH,
    ),
  );
}
