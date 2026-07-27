import 'dart:typed_data';
import 'dart:ui' show Rect;

import 'package:image/image.dart' as img;

/// Detector input is 640x640; embedding input is 256x256.
const int kDetectSize = 640;
const int kEmbedSize = 256;

/// YOLO letterbox padding value (Ultralytics uses 114, /255 on device).
const int kLetterboxPad = 114;

/// Cap the working frame's long side so crop/decode cost and the cross-isolate
/// copy stay bounded. 1280 is plenty of detail for a 256x256 embed.
const int kMaxWorkingSide = 1280;

/// The decoded, upright, downscaled source frame (RGB, 3 bytes/pixel,
/// row-major). Kept so the embed step can crop from ORIGINAL-frame pixels
/// (never from letterbox space).
class DecodedFrame {
  const DecodedFrame({
    required this.width,
    required this.height,
    required this.rgb,
  });

  final int width;
  final int height;
  final Uint8List rgb;
}

/// Output of detector preprocessing: the NCHW tensor + the letterbox geometry
/// needed to invert boxes back to the frame, plus the decoded frame for the
/// crop step.
class DetectBundle {
  const DetectBundle({
    required this.input,
    required this.scale,
    required this.padX,
    required this.padY,
    required this.frame,
  });

  /// Flattened float32 [1,3,640,640], NCHW, RGB, normalized 0..1.
  final Float32List input;

  /// srcPixels * scale + pad = letterbox pixel (forward). Inverse in postproc.
  final double scale;
  final int padX;
  final int padY;
  final DecodedFrame frame;
}

/// Request for the embed preprocessing pass (runs in a [compute] isolate).
class EmbedRequest {
  const EmbedRequest({
    required this.width,
    required this.height,
    required this.rgb,
    required this.box,
    this.margin = 0.06,
  });

  final int width;
  final int height;
  final Uint8List rgb;

  /// Primary box, normalized 0..1 in the frame. Null → center-crop fallback so
  /// the embed step NEVER runs empty.
  final Rect? box;

  /// Fractional margin added around the box before cropping (spec: ~5–8%).
  final double margin;
}

/// The embed tensor plus the pixel crop rect actually used (for the HUD/preview
/// and for tests).
class EmbedBundle {
  const EmbedBundle({required this.input, required this.cropRect});

  /// Flattened float32 [1,3,256,256], NCHW, RGB, 0..1, NO mean/std.
  final Float32List input;

  /// The crop rectangle in original-frame pixels.
  final Rect cropRect;
}

/// Decode a captured JPEG, bake EXIF orientation to upright, downscale to the
/// working cap, and letterbox to the detector's 640x640 input in a single fused
/// pass (resize + /255 + planar NCHW). Top-level so it runs in a [compute]
/// isolate.
///
/// takePicture() returns an already-oriented still, but we bake EXIF defensively
/// (deliberate deviation from PyroGuard's live YUV/BGRA stream: SnapSeek is
/// snap-then-search, so a still JPEG sidesteps per-frame plane wrangling and
/// most orientation traps).
DetectBundle decodeAndPreprocessDetect(Uint8List jpegBytes) {
  img.Image? decoded = img.decodeImage(jpegBytes);
  if (decoded == null) {
    throw const FormatException('Could not decode captured image');
  }
  decoded = img.bakeOrientation(decoded);

  // Downscale so the long side <= kMaxWorkingSide.
  final int longSide =
      decoded.width > decoded.height ? decoded.width : decoded.height;
  if (longSide > kMaxWorkingSide) {
    final double s = kMaxWorkingSide / longSide;
    decoded = img.copyResize(
      decoded,
      width: (decoded.width * s).round(),
      height: (decoded.height * s).round(),
      interpolation: img.Interpolation.average,
    );
  }

  final int srcW = decoded.width;
  final int srcH = decoded.height;

  // Flatten to a compact RGB byte buffer (3 bytes/pixel) once; reused for the
  // detector letterbox below AND shipped back for the crop step.
  final Uint8List rgb = Uint8List(srcW * srcH * 3);
  var p = 0;
  for (var y = 0; y < srcH; y++) {
    for (var x = 0; x < srcW; x++) {
      final px = decoded.getPixel(x, y);
      rgb[p++] = px.r.toInt();
      rgb[p++] = px.g.toInt();
      rgb[p++] = px.b.toInt();
    }
  }

  final Float32List input = buildLetterboxTensor(rgb, srcW, srcH);
  final geom = letterboxGeometry(srcW, srcH);
  return DetectBundle(
    input: input,
    scale: geom.scale,
    padX: geom.padX,
    padY: geom.padY,
    frame: DecodedFrame(width: srcW, height: srcH, rgb: rgb),
  );
}

/// Pure letterbox geometry (scale THEN //2-centered pad) — shared by the
/// forward pass and the tests. Matches the reference Ultralytics recipe.
({double scale, int padX, int padY}) letterboxGeometry(int srcW, int srcH) {
  const size = kDetectSize;
  final double scale =
      (size / srcW) < (size / srcH) ? (size / srcW) : (size / srcH);
  final int newW = (srcW * scale).round();
  final int newH = (srcH * scale).round();
  return (scale: scale, padX: (size - newW) ~/ 2, padY: (size - newH) ~/ 2);
}

/// Build the 640x640 NCHW /255 RGB tensor from a packed RGB buffer in ONE pass
/// over the output grid (nearest sampling), pre-filling pad with 114/255.
Float32List buildLetterboxTensor(Uint8List rgb, int srcW, int srcH) {
  const int size = kDetectSize;
  const int area = size * size;
  const double inv255 = 1.0 / 255.0;
  const double pad = kLetterboxPad * inv255;

  final Float32List input = Float32List(3 * area)..fillRange(0, 3 * area, pad);
  final g = letterboxGeometry(srcW, srcH);
  final int newW = (srcW * g.scale).round();
  final int newH = (srcH * g.scale).round();

  for (var oy = g.padY; oy < g.padY + newH; oy++) {
    if (oy < 0 || oy >= size) continue;
    final int uy = ((oy - g.padY) / g.scale).floor().clamp(0, srcH - 1);
    final int rowBase = oy * size;
    final int srcRow = uy * srcW * 3;
    for (var ox = g.padX; ox < g.padX + newW; ox++) {
      if (ox < 0 || ox >= size) continue;
      final int ux = ((ox - g.padX) / g.scale).floor().clamp(0, srcW - 1);
      final int si = srcRow + ux * 3;
      final int p = rowBase + ox;
      input[p] = rgb[si] * inv255;
      input[area + p] = rgb[si + 1] * inv255;
      input[2 * area + p] = rgb[si + 2] * inv255;
    }
  }
  return input;
}

/// Map a normalized box (or the center-crop fallback) to a pixel crop rect.
/// Public + pure so the crop-space test can assert it directly.
Rect cropRectFor(EmbedRequest req) {
  final int w = req.width;
  final int h = req.height;
  final Rect? box = req.box;

  double x1, y1, x2, y2;
  if (box == null) {
    // Center square crop (fallback when the detector found nothing).
    final double side = (w < h ? w : h).toDouble();
    final double cx = w / 2, cy = h / 2;
    x1 = cx - side / 2;
    y1 = cy - side / 2;
    x2 = cx + side / 2;
    y2 = cy + side / 2;
  } else {
    x1 = box.left * w;
    y1 = box.top * h;
    x2 = box.right * w;
    y2 = box.bottom * h;
    // Expand by the fractional margin around the box.
    final double mx = (x2 - x1) * req.margin;
    final double my = (y2 - y1) * req.margin;
    x1 -= mx;
    y1 -= my;
    x2 += mx;
    y2 += my;
  }

  x1 = x1.clamp(0.0, w.toDouble());
  y1 = y1.clamp(0.0, h.toDouble());
  x2 = x2.clamp(0.0, w.toDouble());
  y2 = y2.clamp(0.0, h.toDouble());
  if (x2 <= x1) {
    x1 = 0;
    x2 = w.toDouble();
  }
  if (y2 <= y1) {
    y1 = 0;
    y2 = h.toDouble();
  }
  return Rect.fromLTRB(x1, y1, x2, y2);
}

/// Crop the frame to the primary box (+margin, from ORIGINAL-frame pixels) and
/// resize to 256x256, /255, RGB, NCHW — NO ImageNet mean/std (MobileCLIP uses
/// plain [0,1]). Top-level so it runs in a [compute] isolate.
EmbedBundle preprocessEmbed(EmbedRequest req) {
  final Rect crop = cropRectFor(req);
  final int cx0 = crop.left.floor();
  final int cy0 = crop.top.floor();
  final int cw = (crop.width).round().clamp(1, req.width - cx0);
  final int ch = (crop.height).round().clamp(1, req.height - cy0);

  const int out = kEmbedSize;
  const int area = out * out;
  const double inv255 = 1.0 / 255.0;
  final Float32List input = Float32List(3 * area);

  // Bilinear resize the crop region directly into planar NCHW in one pass.
  final double sx = cw / out;
  final double sy = ch / out;
  final int rowStride = req.width * 3;
  for (var oy = 0; oy < out; oy++) {
    final double fy = (oy + 0.5) * sy - 0.5;
    int y0 = fy.floor();
    double wy = fy - y0;
    if (y0 < 0) {
      y0 = 0;
      wy = 0;
    }
    int y1 = y0 + 1;
    if (y1 > ch - 1) {
      y1 = ch - 1;
      if (y0 > y1) y0 = y1;
    }
    final int ay0 = (cy0 + y0) * rowStride;
    final int ay1 = (cy0 + y1) * rowStride;
    final int orow = oy * out;
    for (var ox = 0; ox < out; ox++) {
      final double fx = (ox + 0.5) * sx - 0.5;
      int x0 = fx.floor();
      double wx = fx - x0;
      if (x0 < 0) {
        x0 = 0;
        wx = 0;
      }
      int x1 = x0 + 1;
      if (x1 > cw - 1) {
        x1 = cw - 1;
        if (x0 > x1) x0 = x1;
      }
      final int i00 = ay0 + (cx0 + x0) * 3;
      final int i10 = ay0 + (cx0 + x1) * 3;
      final int i01 = ay1 + (cx0 + x0) * 3;
      final int i11 = ay1 + (cx0 + x1) * 3;
      final int p = orow + ox;
      for (var c = 0; c < 3; c++) {
        final double top =
            req.rgb[i00 + c] + (req.rgb[i10 + c] - req.rgb[i00 + c]) * wx;
        final double bot =
            req.rgb[i01 + c] + (req.rgb[i11 + c] - req.rgb[i01 + c]) * wx;
        input[c * area + p] = (top + (bot - top) * wy) * inv255;
      }
    }
  }
  return EmbedBundle(input: input, cropRect: crop);
}
