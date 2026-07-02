import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:zetic_mlange/zetic_mlange.dart';

import 'letterbox.dart';

/// Result of preprocessing one still image: the NCHW input buffer plus the
/// exact letterbox transform used (needed to un-letterbox the model's boxes)
/// and the decoded original-image dimensions (needed for the display fit).
class PreprocessResult {
  const PreprocessResult({
    required this.input,
    required this.transform,
    required this.originalWidth,
    required this.originalHeight,
  });

  /// Flattened `float32[1,3,640,640]`, NCHW, RGB, values 0..1.
  final Float32List input;
  final LetterboxTransform transform;
  final int originalWidth;
  final int originalHeight;

  /// Wrap as the SDK input tensor bound to `images`.
  Tensor toTensor() =>
      Tensor.float32List(input, shape: const [1, 3, Preprocessor.inputSize, Preprocessor.inputSize]);
}

/// Turns selected image bytes into the model's input tensor, exactly matching
/// validate_demo.py: decode -> EXIF orient -> letterbox 640 (gray 114 pad) ->
/// RGB -> /255 -> NCHW.
class Preprocessor {
  const Preprocessor();

  static const int inputSize = 640;
  static const int padValue = 114; // gray letterbox pad (~0.447 after /255)

  /// Decode encoded bytes (JPEG/PNG/…), apply EXIF orientation, then process.
  PreprocessResult process(Uint8List encodedBytes) {
    final decoded = img.decodeImage(encodedBytes);
    if (decoded == null) {
      throw const FormatException('Could not decode the selected image.');
    }
    // Honor EXIF orientation so the image is upright before letterboxing.
    final oriented = img.bakeOrientation(decoded);
    return processDecoded(oriented);
  }

  /// Process an already-decoded, already-upright image.
  PreprocessResult processDecoded(img.Image oriented) {
    final w = oriented.width;
    final h = oriented.height;
    final transform = LetterboxTransform.compute(
      originalWidth: w,
      originalHeight: h,
      targetSize: inputSize,
    );

    // Resize preserving aspect (linear ~ cv2 INTER_LINEAR).
    final resized = img.copyResize(
      oriented,
      width: transform.resizedWidth,
      height: transform.resizedHeight,
      interpolation: img.Interpolation.linear,
    );

    // Center-pad onto a 640x640 gray(114) canvas.
    final canvas = img.Image(width: inputSize, height: inputSize, numChannels: 3);
    img.fill(canvas, color: img.ColorRgb8(padValue, padValue, padValue));
    img.compositeImage(
      canvas,
      resized,
      dstX: transform.padX,
      dstY: transform.padY,
    );

    return PreprocessResult(
      input: _toNchwFloat32(canvas),
      transform: transform,
      originalWidth: w,
      originalHeight: h,
    );
  }

  /// RGB interleaved bytes -> planar NCHW float32 in [0,1], single pass.
  static Float32List _toNchwFloat32(img.Image canvas) {
    final rgb = canvas.getBytes(order: img.ChannelOrder.rgb); // len = 640*640*3
    const plane = inputSize * inputSize;
    final out = Float32List(3 * plane);
    var px = 0;
    for (var i = 0; i < plane; i++, px += 3) {
      out[i] = rgb[px] / 255.0; // R plane
      out[plane + i] = rgb[px + 1] / 255.0; // G plane
      out[2 * plane + i] = rgb[px + 2] / 255.0; // B plane
    }
    return out;
  }
}
