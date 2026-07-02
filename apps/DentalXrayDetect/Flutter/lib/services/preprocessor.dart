import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// Model input side length. **640** for this YOLO11n dental export (NOT the 928
/// some sibling apps use). A stray 640-vs-928 anywhere (here, the NCHW buffer,
/// or the letterbox inverse) silently misplaces every box.
const int kInputSize = 640;

/// Number of float elements in one NCHW input tensor: 1 * 3 * 640 * 640.
const int kInputElements = 3 * kInputSize * kInputSize;

/// Letterbox pad color, normalized. 114/255 is the standard Ultralytics gray
/// (0.447 after ÷255) — the exact value the Python validation harness pads with.
const double kPadValue = 114.0 / 255.0;

/// Geometry of a letterbox-resize from a source frame into a square [target].
///
/// Aspect ratio is preserved; the scaled image is centered with symmetric
/// padding. The forward map (used by the fused preprocess) and the inverse map
/// (used by the postprocess) are exact reverses of each other. `r`, `padX`,
/// `padY` are recorded here and consumed verbatim by the inverse.
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

  /// Inverse map: a model-space (letterboxed) X back to original-image X.
  double modelToSrcX(double mx) => (mx - padX) / scale;

  /// Inverse map: a model-space (letterboxed) Y back to original-image Y.
  double modelToSrcY(double my) => (my - padY) / scale;
}

/// Compute letterbox geometry for a [srcW] x [srcH] frame into [target]x[target].
/// Mirrors the harness `r = min(new/h, new/w)` with centered padding.
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

/// Bilinear resize + center-pad + normalize + NCHW reorder for a packed RGB
/// source.
///
/// The scale step uses **bilinear** interpolation (`img.copyResize` with
/// `Interpolation.linear`) to reproduce the validated harness / Ultralytics
/// letterbox (`cv2.INTER_LINEAR`) — matching bilinear on-device keeps
/// confidences (and therefore borderline detections at the 0.45 gate) aligned
/// with the measured demo recall. Nearest-neighbour would shift confidences and
/// could flip borderline boxes.
///
/// Flow: reconstruct an [img.Image] from [rgb], bilinear-resize to the scaled
/// region ([LetterboxParams.scaledW] x [scaledH]), then place it centered with
/// gray ([kPadValue]) padding into the preallocated NCHW [out] buffer,
/// normalizing to 0..1. The center offset uses `floor` to mirror the harness's
/// integer `dw = (640 - nw) // 2` placement. Box *placement* is governed by the
/// letterbox geometry, unchanged.
///
/// This is a STILL-image app (one inference per upload), so the extra resize
/// buffer is a per-upload cost, not a per-frame hot-loop cost — fidelity is the
/// right trade here. [rgb] is row-major, 3 bytes/pixel, [srcW]*[srcH]*3 long. A
/// grayscale radiograph decoded to RGB already has its luma replicated across
/// R/G/B. [out] must be length [kInputElements] and is reused across inferences.
void letterboxRgbToNchw(
  Uint8List rgb,
  int srcW,
  int srcH,
  LetterboxParams p,
  Float32List out,
) {
  const int size = kInputSize;
  const int channelStride = size * size; // H*W per channel

  // Bilinear resize the source into the scaled region (harness-matching).
  final img.Image src = img.Image.fromBytes(
    width: srcW,
    height: srcH,
    bytes: rgb.buffer,
    numChannels: 3,
    order: img.ChannelOrder.rgb,
  );
  final img.Image resized = img.copyResize(
    src,
    width: p.scaledW,
    height: p.scaledH,
    interpolation: img.Interpolation.linear,
  );
  final Uint8List rb = resized.getBytes(order: img.ChannelOrder.rgb);

  final int offX = p.padX.floor();
  final int offY = p.padY.floor();
  final int rW = p.scaledW;
  final int rH = p.scaledH;

  for (int dy = 0; dy < size; dy++) {
    final int ry = dy - offY;
    final bool rowInside = ry >= 0 && ry < rH;
    final int rbRow = ry * rW * 3;
    final int rowBase = dy * size;
    for (int dx = 0; dx < size; dx++) {
      final int rIdx = rowBase + dx;
      final int gIdx = channelStride + rIdx;
      final int bIdx = 2 * channelStride + rIdx;
      final int rx = dx - offX;
      if (rowInside && rx >= 0 && rx < rW) {
        final int off = rbRow + rx * 3;
        out[rIdx] = rb[off] / 255.0;
        out[gIdx] = rb[off + 1] / 255.0;
        out[bIdx] = rb[off + 2] / 255.0;
      } else {
        out[rIdx] = kPadValue;
        out[gIdx] = kPadValue;
        out[bIdx] = kPadValue;
      }
    }
  }
}
