import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

/// A decoded, orientation-baked radiograph in two packed layouts:
///  * [rgb]  — 3 bytes/pixel, fed straight into the 640 letterbox (the model).
///  * [rgba] — 4 bytes/pixel, used to build a `ui.Image` for on-screen display.
///
/// [width]/[height] are the ORIENTED dimensions and are the single source of
/// truth for the whole still path: the letterbox, the returned box coordinates,
/// and the display all share this pixel space, so overlay boxes cannot drift.
class DecodedImage {
  const DecodedImage({
    required this.rgb,
    required this.rgba,
    required this.width,
    required this.height,
  });

  final Uint8List rgb;
  final Uint8List rgba;
  final int width;
  final int height;
}

/// Longest side we keep. Panoramic radiographs are large (~2872x1504+); the
/// model only ever sees 640, so capping here bounds preprocessing cost and the
/// isolate copy without affecting detections. Coordinates stay consistent
/// because we report the capped dimensions everywhere.
const int kMaxStillSide = 2048;

/// Decode picked/bundled image [fileBytes] off the UI isolate: parse, apply the
/// EXIF orientation, optionally downscale, and extract packed RGB + RGBA.
///
/// A grayscale radiograph is decoded straight to 3-channel RGB (luma replicated
/// across R/G/B), which is exactly what the model expects. Returns null if the
/// bytes aren't a decodable image.
Future<DecodedImage?> decodeStillImage(Uint8List fileBytes) =>
    compute(_decodeStillImage, fileBytes);

DecodedImage? _decodeStillImage(Uint8List fileBytes) {
  final img.Image? decoded = img.decodeImage(fileBytes);
  if (decoded == null) return null;

  // Respect EXIF orientation (phones store a rotation flag; a photo of a mounted
  // X-ray may carry one). This is the ONLY orientation concern — there is no
  // live camera buffer, so no rotating-buffer trap.
  img.Image oriented = img.bakeOrientation(decoded);

  final int longest =
      oriented.width > oriented.height ? oriented.width : oriented.height;
  if (longest > kMaxStillSide) {
    if (oriented.width >= oriented.height) {
      oriented = img.copyResize(oriented, width: kMaxStillSide);
    } else {
      oriented = img.copyResize(oriented, height: kMaxStillSide);
    }
  }

  return DecodedImage(
    rgb: oriented.getBytes(order: img.ChannelOrder.rgb),
    rgba: oriented.getBytes(order: img.ChannelOrder.rgba),
    width: oriented.width,
    height: oriented.height,
  );
}
