import 'dart:typed_data';

import 'package:image/image.dart' as img;

import 'frame_data.dart';
import 'plate_ocr.dart' show CropRect, computeCropRect;

/// A decoded still photo, packed as an upright BGRA buffer so it feeds the
/// EXISTING pipeline unchanged: the same [FrameData] the live camera builds
/// (format bgra8888, rotation 0) drives the same [Preprocessor]/model, and the
/// same BGRA buffer feeds [PlateOcr.recognize]. No pipeline is duplicated.
class StillImage {
  StillImage({
    required this.bgra,
    required this.width,
    required this.height,
  });

  /// Tightly-packed BGRA8888 pixels (row stride == width*4), upright.
  final Uint8List bgra;
  final int width;
  final int height;

  int get bytesPerRow => width * 4;

  /// Build the isolate-sendable frame the [MelangeService] expects. Rotation 0:
  /// a decoded still is already upright (EXIF baked in [decodeStillImage]).
  FrameData toFrameData() => FrameData(
        format: FramePixelFormat.bgra8888,
        width: width,
        height: height,
        rotationDegrees: 0,
        plane0: bgra,
        bytesPerRow0: bytesPerRow,
      );
}

/// Longest-side cap applied to a picked photo before detection. The model only
/// consumes 640x640, so a huge original wastes memory across the isolate copy;
/// this keeps enough resolution for OCR while bounding cost. Kept generous.
const int kStillMaxSide = 1920;

/// Pure: downscale (w,h) so its longer side is at most [maxSide], preserving
/// aspect. Never upscales; never returns a zero dimension.
(int, int) scaledDimensions(int w, int h, int maxSide) {
  final longSide = w > h ? w : h;
  if (longSide <= 0 || longSide <= maxSide) return (w, h);
  final s = maxSide / longSide;
  final nw = (w * s).round();
  final nh = (h * s).round();
  return (nw < 1 ? 1 : nw, nh < 1 ? 1 : nh);
}

/// Decode encoded image bytes (JPEG/PNG/HEIC-as-decoded) into an upright
/// [StillImage]. Bakes EXIF orientation so the pixel grid matches how Flutter's
/// `Image` widget displays the same file. Returns null on undecodable input.
///
/// Top-level so it can run via `compute` off the UI isolate (decode is heavy).
StillImage? decodeStillImage(Uint8List encoded) {
  final decoded = img.decodeImage(encoded);
  if (decoded == null) return null;
  final oriented = img.bakeOrientation(decoded);
  final (w, h) = scaledDimensions(oriented.width, oriented.height, kStillMaxSide);
  final resized = (w == oriented.width && h == oriented.height)
      ? oriented
      : img.copyResize(oriented, width: w, height: h);
  final bgra = resized.getBytes(order: img.ChannelOrder.bgra);
  return StillImage(bgra: bgra, width: resized.width, height: resized.height);
}

/// Short side (px) a plate crop is upscaled to so the user can actually READ the
/// plate. Small/distant plates are only a few dozen pixels tall in the source;
/// enlarging the isolated crop is what makes the OCR verifiable by eye.
const int kPlateCropMinSide = 160;

/// Crop the detection region [left,top,right,bottom] (upright source-image
/// pixels) out of [still]'s BGRA buffer, upscale it so its short side is at
/// least [kPlateCropMinSide], and encode a PNG. The region is padded + clamped
/// via [computeCropRect] for a little context around the plate. Returns null for
/// a degenerate/out-of-bounds region.
///
/// This is the "readable plate verification" primitive: each list row shows the
/// zoomed crop NEXT TO the OCR text so the plate can be compared by eye.
Uint8List? encodePlateCropPng(
  StillImage still,
  double left,
  double top,
  double right,
  double bottom, {
  double padFraction = 0.08,
}) {
  final CropRect r = computeCropRect(
    left,
    top,
    right,
    bottom,
    still.width,
    still.height,
    padFraction: padFraction,
  );
  if (r.isEmpty) return null;

  final crop = img.Image(width: r.width, height: r.height);
  final stride = still.bytesPerRow;
  for (var y = 0; y < r.height; y++) {
    var s = (r.y + y) * stride + r.x * 4;
    for (var x = 0; x < r.width; x++) {
      // BGRA -> RGB.
      crop.setPixelRgb(x, y, still.bgra[s + 2], still.bgra[s + 1], still.bgra[s]);
      s += 4;
    }
  }

  final shortSide = r.width < r.height ? r.width : r.height;
  final scaled = shortSide >= kPlateCropMinSide
      ? crop
      : img.copyResize(
          crop,
          width: (r.width * kPlateCropMinSide / shortSide).round(),
          interpolation: img.Interpolation.nearest,
        );
  return img.encodePng(scaled);
}
