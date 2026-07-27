import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// Pure-Dart pre-processing for the Marqo ViT-Tiny NSFW/SFW classifier.
///
/// This reproduces the timm eval transform EXACTLY — the #1 silent-wrong trap
/// for this app (SPEC_STUB.md "Pre-processing pipeline" / "Validation focus"):
///
///   1. apply EXIF orientation, decode to RGB (drop alpha)
///   2. resize so the SHORTEST edge = 384, BICUBIC with ANTIALIASING, preserve aspect
///   3. center-crop the middle 384 x 384
///   4. scale /255              -> [0, 1]
///   5. normalize (v - 0.5)/0.5 (mean = std = 0.5 per channel) -> [-1, 1]
///   6. reorder HWC -> NCHW [1, 3, 384, 384], RGB channel order
///
/// It is NOT a squash-resize, NOT plain /255, NOT ImageNet mean/std, and — the
/// subtle one — NOT a plain bicubic. A downscale needs an ANTIALIASED bicubic
/// (the resampling filter's support is scaled by the downscale factor, exactly
/// like Pillow / timm `antialias=True`). A point-sampled bicubic aliases and
/// measurably shifts borderline scores. That resampler is implemented here.
///
/// Reference: the identical float algorithm is in `test/fixtures/gen_golden.py`,
/// which generates the golden tensors the resize-fidelity test checks against.
class Preprocessor {
  const Preprocessor._();

  /// Final square input side fed to the model.
  static const int inputSize = 384;

  /// Number of float32 elements in one input tensor (3 * 384 * 384).
  static const int tensorLength = 3 * inputSize * inputSize;

  /// The input tensor shape bound to `pixel_values`.
  static const List<int> tensorShape = [1, 3, inputSize, inputSize];

  /// Keys cubic-convolution parameter (a = -0.5), matching Pillow / timm.
  static const double _a = -0.5;

  /// Cubic filter support radius (in input pixels, before downscale scaling).
  static const double _support = 2.0;

  /// Normalize one 0–255 channel value to [-1, 1]: `(v/255 - 0.5) / 0.5`.
  /// Equivalent to `v/127.5 - 1`. Exposed for unit-testing the exact formula.
  static double normalizePixel(num value255) => (value255 / 255.0 - 0.5) / 0.5;

  /// Keys cubic-convolution kernel with a = -0.5. Exposed for unit tests.
  static double cubicKernel(double x) {
    x = x.abs();
    if (x < 1.0) return ((_a + 2.0) * x - (_a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * _a) * x + 8.0 * _a) * x - 4.0 * _a;
    return 0.0;
  }

  /// New (width, height) after scaling the shortest edge to [inputSize] while
  /// preserving aspect ratio. Uses truncation (floor), matching the reference.
  static (int width, int height) resizedDimensions(int width, int height) {
    final shortest = width < height ? width : height;
    final scale = inputSize / shortest;
    return ((width * scale).toInt(), (height * scale).toInt());
  }

  /// Top-left origin of the centered [inputSize] x [inputSize] crop (floor).
  static (int x, int y) cropOrigin(int width, int height) =>
      ((width - inputSize) ~/ 2, (height - inputSize) ~/ 2);

  /// Decode raw image bytes, honoring EXIF orientation, into an RGB image.
  static img.Image decodeOriented(Uint8List bytes) {
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw const FormatException('Could not decode the selected image.');
    }
    return img.bakeOrientation(decoded);
  }

  /// Precompute the resampling coefficients for output indices `[start, end)`
  /// of a 1-D axis being resized from [inSize] to [outSize].
  ///
  /// Mirrors Pillow's `precompute_coeffs`: for a downscale the filter support is
  /// stretched by the scale factor (antialiasing); coefficients are clamped at
  /// the axis edges and normalized to sum to 1. Only the requested output range
  /// is computed (the rest is cropped away) — numerically identical to computing
  /// the full axis and cropping, since each output pixel is independent.
  static (List<int> starts, List<Float32List> weights) _coeffs(
    int inSize,
    int outSize,
    int start,
    int end,
  ) {
    final scale = inSize / outSize;
    final filterScale = scale < 1.0 ? 1.0 : scale;
    final support = _support * filterScale;
    final invFilterScale = 1.0 / filterScale;

    final starts = <int>[];
    final weights = <Float32List>[];
    for (var xx = start; xx < end; xx++) {
      final center = (xx + 0.5) * scale;
      var xmin = (center - support + 0.5).toInt();
      if (xmin < 0) xmin = 0;
      var xmax = (center + support + 0.5).toInt();
      if (xmax > inSize) xmax = inSize;
      final n = xmax - xmin;
      final k = Float32List(n);
      var sum = 0.0;
      for (var i = 0; i < n; i++) {
        final w = cubicKernel((i + xmin - center + 0.5) * invFilterScale);
        k[i] = w;
        sum += w;
      }
      if (sum != 0.0) {
        for (var i = 0; i < n; i++) {
          k[i] = k[i] / sum;
        }
      }
      starts.add(xmin);
      weights.add(k);
    }
    return (starts, weights);
  }

  /// Run resize (antialiased bicubic) -> center-crop -> normalize -> NCHW on a
  /// decoded image and return the flattened `float32[1,3,384,384]` input data.
  static Float32List imageToTensor(img.Image source) {
    final srcW = source.width;
    final srcH = source.height;
    final (newW, newH) = resizedDimensions(srcW, srcH);
    final (cropX, cropY) = cropOrigin(newW, newH);

    // Source pixels as a flat RGB byte buffer (drops any alpha channel).
    final rgb = source.getBytes(order: img.ChannelOrder.rgb);

    // --- Horizontal pass: resize width srcW -> newW, but only the 384 output
    // columns that survive the center crop. Produces horiz[srcH][inputSize][3].
    final (hStarts, hWeights) = _coeffs(srcW, newW, cropX, cropX + inputSize);
    final horiz = Float32List(srcH * inputSize * 3);
    final srcRowStride = srcW * 3;
    for (var y = 0; y < srcH; y++) {
      final srcRow = y * srcRowStride;
      final dstRow = y * inputSize * 3;
      for (var ox = 0; ox < inputSize; ox++) {
        final k = hWeights[ox];
        final n = k.length;
        var r = 0.0, g = 0.0, b = 0.0;
        var sp = srcRow + hStarts[ox] * 3;
        for (var i = 0; i < n; i++) {
          final w = k[i];
          r += rgb[sp] * w;
          g += rgb[sp + 1] * w;
          b += rgb[sp + 2] * w;
          sp += 3;
        }
        // Clamp to [0,255] after the pass, matching Pillow's 8-bit intermediate;
        // the cubic filter's negative lobes overshoot on sharp edges.
        final dp = dstRow + ox * 3;
        horiz[dp] = _clamp255(r);
        horiz[dp + 1] = _clamp255(g);
        horiz[dp + 2] = _clamp255(b);
      }
    }

    // --- Vertical pass: resize height srcH -> newH, only the 384 surviving rows.
    // Fuse /255 + (x-0.5)/0.5 + HWC->NCHW placement into this single write.
    final (vStarts, vWeights) = _coeffs(srcH, newH, cropY, cropY + inputSize);
    final data = Float32List(tensorLength);
    const plane = inputSize * inputSize;
    final horizRowStride = inputSize * 3;
    for (var oy = 0; oy < inputSize; oy++) {
      final k = vWeights[oy];
      final n = k.length;
      final rowBase = oy * inputSize;
      final colBase = vStarts[oy] * horizRowStride;
      for (var ox = 0; ox < inputSize; ox++) {
        var r = 0.0, g = 0.0, b = 0.0;
        var sp = colBase + ox * 3;
        for (var i = 0; i < n; i++) {
          final w = k[i];
          r += horiz[sp] * w;
          g += horiz[sp + 1] * w;
          b += horiz[sp + 2] * w;
          sp += horizRowStride;
        }
        // Clamp the vertical-pass result to [0,255] (Pillow parity), then fuse
        // /255 + (x-0.5)/0.5 into the NCHW write.
        final idx = rowBase + ox;
        data[idx] = (_clamp255(r) / 255.0 - 0.5) / 0.5; // R plane (channel 0)
        data[plane + idx] = (_clamp255(g) / 255.0 - 0.5) / 0.5; // G plane
        data[2 * plane + idx] = (_clamp255(b) / 255.0 - 0.5) / 0.5; // B plane
      }
    }
    return data;
  }

  /// Clamp a resampled channel value to the valid 8-bit range [0, 255].
  static double _clamp255(double v) => v < 0.0 ? 0.0 : (v > 255.0 ? 255.0 : v);

  /// Full pipeline: raw file bytes -> model input data. Runs entirely in Dart.
  static Float32List preprocess(Uint8List bytes) =>
      imageToTensor(decodeOriented(bytes));
}

/// Top-level entry point usable with `compute()` to keep image decode +
/// resampling off the UI isolate. Returns the flattened `float32[1,3,384,384]`.
Float32List preprocessImageBytes(Uint8List bytes) =>
    Preprocessor.preprocess(bytes);

/// The preprocessed tensor plus the decoded (EXIF-baked) buffer size, for the
/// on-screen diagnostics HUD. Sendable across the `compute()` isolate boundary.
class PreprocessOutput {
  const PreprocessOutput(this.data, this.width, this.height);

  final Float32List data;
  final int width;
  final int height;
}

/// `compute()` entry point that also reports the decoded buffer WxH (surfaced on
/// the HUD, since Dart `print` is invisible in a release device build).
PreprocessOutput preprocessWithDims(Uint8List bytes) {
  final image = Preprocessor.decodeOriented(bytes);
  final data = Preprocessor.imageToTensor(image);
  return PreprocessOutput(data, image.width, image.height);
}
