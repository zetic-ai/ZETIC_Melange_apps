import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// Pure-Dart pre-processing for the MobileNetV2 DR screener.
///
/// This reproduces the model's HF `MobileNetV2ImageProcessor` pipeline EXACTLY —
/// the #1 silent-wrong trap for this app (SPEC.md "Pre-processing pipeline"):
///
///   1. apply EXIF orientation, decode to RGB (drop alpha)
///   2. resize SHORTEST edge -> 256 (bilinear), preserve aspect
///   3. center-crop the middle 224 x 224
///   4. scale /255            -> [0, 1]
///   5. normalize (v - 0.5)/0.5 (mean = std = 0.5 per channel) -> [-1, 1]
///   6. reorder HWC -> NCHW [1, 3, 224, 224], RGB channel order
///
/// It is NOT a squash-resize to 224, NOT plain /255, and NOT ImageNet mean/std.
class Preprocessor {
  const Preprocessor._();

  /// Final square input side fed to the model.
  static const int inputSize = 224;

  /// Shortest edge the image is resized to before the center-crop.
  static const int resizeShortestEdge = 256;

  /// Number of float32 elements in one input tensor (3 * 224 * 224).
  static const int tensorLength = 3 * inputSize * inputSize;

  /// The input tensor shape bound to `pixel_values`.
  static const List<int> tensorShape = [1, 3, inputSize, inputSize];

  /// Normalize one 0–255 channel value to [-1, 1]: `(v/255 - 0.5) / 0.5`.
  ///
  /// Equivalent to `v/127.5 - 1`. Exposed for unit testing the exact formula.
  static double normalizePixel(num value255) =>
      (value255 / 255.0 - 0.5) / 0.5;

  /// New (width, height) after scaling the shortest edge to [resizeShortestEdge]
  /// while preserving aspect ratio (matching the crop that follows).
  static (int width, int height) resizedDimensions(int width, int height) {
    final shortest = width < height ? width : height;
    final scale = resizeShortestEdge / shortest;
    final newW = (width * scale).round();
    final newH = (height * scale).round();
    return (newW, newH);
  }

  /// Top-left origin of the centered [inputSize] x [inputSize] crop.
  static (int x, int y) cropOrigin(int width, int height) {
    final x = ((width - inputSize) / 2).floor();
    final y = ((height - inputSize) / 2).floor();
    return (x, y);
  }

  /// Decode raw image bytes, honoring EXIF orientation, into an RGB image.
  ///
  /// [img.bakeOrientation] physically applies any EXIF rotation flag so the
  /// retina is upright before geometry. Throws if the bytes cannot be decoded.
  static img.Image decodeOriented(Uint8List bytes) {
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw const FormatException('Could not decode the selected image.');
    }
    return img.bakeOrientation(decoded);
  }

  /// Run the resize -> center-crop -> normalize -> NCHW pipeline on a decoded
  /// image and return the flattened `float32[1,3,224,224]` input data.
  static Float32List imageToTensor(img.Image source) {
    final (newW, newH) = resizedDimensions(source.width, source.height);
    final resized = img.copyResize(
      source,
      width: newW,
      height: newH,
      interpolation: img.Interpolation.linear, // bilinear
    );

    final (cropX, cropY) = cropOrigin(resized.width, resized.height);
    final cropped = img.copyCrop(
      resized,
      x: cropX,
      y: cropY,
      width: inputSize,
      height: inputSize,
    );

    // Pull a flat RGB byte buffer once (this also drops any alpha channel).
    // Iterating this typed buffer is ~11% faster than per-pixel getPixel(),
    // which allocates a Pixel accessor per call.
    final rgb = cropped.getBytes(order: img.ChannelOrder.rgb);

    // NCHW, RGB. Fused single pass: read each pixel once, write into the three
    // channel planes with the normalization applied in place.
    final data = Float32List(tensorLength);
    const plane = inputSize * inputSize;
    var src = 0;
    for (var i = 0; i < plane; i++) {
      data[i] = normalizePixel(rgb[src]); // R plane (channel 0)
      data[plane + i] = normalizePixel(rgb[src + 1]); // G plane (channel 1)
      data[2 * plane + i] = normalizePixel(rgb[src + 2]); // B plane (channel 2)
      src += 3;
    }
    return data;
  }

  /// Full pipeline: raw file bytes -> model input data. Runs entirely in Dart.
  static Float32List preprocess(Uint8List bytes) =>
      imageToTensor(decodeOriented(bytes));
}

/// Top-level entry point usable with `compute()` to keep image decode off the
/// UI isolate. Returns the flattened `float32[1,3,224,224]` input data.
Float32List preprocessFundusBytes(Uint8List bytes) =>
    Preprocessor.preprocess(bytes);
