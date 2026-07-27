import 'dart:io';
import 'dart:typed_data';

import 'package:contentmoderation/services/preprocessor.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

/// Tier A — the resize-interpolation trap. The pipeline MUST reproduce an
/// ANTIALIASED bicubic resize (timm `antialias=True`), not a plain bicubic and
/// not bilinear/nearest — the latter measurably shift borderline scores.
///
/// Strategy: `test/fixtures/gen_golden.py` implements the SAME float separable
/// cubic-antialias resampler in numpy and writes golden NCHW tensors for lossless
/// PNG fixtures. PNG decode is identical between PIL and Dart's image package, so
/// the Dart pipeline must match the golden to float epsilon (the only residual is
/// double- vs float32 accumulation order). A second check proves the resize is
/// genuinely antialiased by showing it differs from a plain bilinear resize.
Float32List loadGolden(String name) {
  final bytes = File('test/fixtures/golden/$name.f32').readAsBytesSync();
  final bd = ByteData.sublistView(bytes);
  final out = Float32List(Preprocessor.tensorLength);
  for (var i = 0; i < out.length; i++) {
    out[i] = bd.getFloat32(i * 4, Endian.little);
  }
  return out;
}

({double maxAbs, double meanAbs}) diff(Float32List a, Float32List b) {
  var maxAbs = 0.0;
  var sum = 0.0;
  for (var i = 0; i < a.length; i++) {
    final d = (a[i] - b[i]).abs();
    if (d > maxAbs) maxAbs = d;
    sum += d;
  }
  return (maxAbs: maxAbs, meanAbs: sum / a.length);
}

void main() {
  const fixtures = ['grad', 'hifreq', 'tall'];

  group('Dart cubic-antialias pipeline reproduces the PIL-formula golden', () {
    for (final name in fixtures) {
      test('$name.png -> golden/$name.f32 within tolerance', () {
        final bytes = File('test/fixtures/$name.png').readAsBytesSync();
        final image = Preprocessor.decodeOriented(bytes);
        final got = Preprocessor.imageToTensor(image);
        final golden = loadGolden(name);
        expect(got.length, golden.length);
        final d = diff(got, golden);
        // ignore: avoid_print
        print('resize-fidelity $name: maxAbs=${d.maxAbs.toStringAsFixed(6)} '
            'meanAbs=${d.meanAbs.toStringAsFixed(7)}');
        // Lossless PNG + identical algorithm: match to float epsilon.
        expect(d.meanAbs, lessThan(1e-4));
        expect(d.maxAbs, lessThan(2e-3));
      });
    }
  });

  group('the resize is genuinely ANTIALIASED (differs from bilinear)', () {
    test('cubic-antialias output differs measurably from a bilinear resize', () {
      final bytes = File('test/fixtures/hifreq.png').readAsBytesSync();
      final image = Preprocessor.decodeOriented(bytes);
      final cubic = Preprocessor.imageToTensor(image);

      // A naive bilinear resize (image pkg linear) -> center-crop -> normalize.
      final (nw, nh) =
          Preprocessor.resizedDimensions(image.width, image.height);
      final resized = img.copyResize(image,
          width: nw, height: nh, interpolation: img.Interpolation.linear);
      final (cx, cy) = Preprocessor.cropOrigin(nw, nh);
      final cropped = img.copyCrop(resized,
          x: cx, y: cy, width: Preprocessor.inputSize,
          height: Preprocessor.inputSize);
      final rgb = cropped.getBytes(order: img.ChannelOrder.rgb);
      final bilinear = Float32List(Preprocessor.tensorLength);
      const plane = Preprocessor.inputSize * Preprocessor.inputSize;
      var src = 0;
      for (var i = 0; i < plane; i++) {
        bilinear[i] = Preprocessor.normalizePixel(rgb[src]);
        bilinear[plane + i] = Preprocessor.normalizePixel(rgb[src + 1]);
        bilinear[2 * plane + i] = Preprocessor.normalizePixel(rgb[src + 2]);
        src += 3;
      }

      final d = diff(cubic, bilinear);
      // ignore: avoid_print
      print('cubic-vs-bilinear on hifreq: maxAbs=${d.maxAbs.toStringAsFixed(4)} '
          'meanAbs=${d.meanAbs.toStringAsFixed(5)}');
      // If the pipeline had silently fallen back to bilinear, this would be ~0.
      expect(d.meanAbs, greaterThan(1e-3));
    });
  });
}
