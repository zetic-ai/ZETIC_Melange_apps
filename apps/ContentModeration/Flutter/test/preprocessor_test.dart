import 'package:contentmoderation/services/preprocessor.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

/// Helper: index into the NCHW [1,3,384,384] flat buffer.
int nchw(int c, int y, int x) =>
    c * Preprocessor.inputSize * Preprocessor.inputSize +
    y * Preprocessor.inputSize +
    x;

/// Tier A — normalization exactness, channel order, and resize/crop geometry.
void main() {
  group('(v-0.5)/0.5 normalization exactness (NOT plain /255, NOT ImageNet)', () {
    test('maps 0 -> -1, 127.5 -> 0, 255 -> +1', () {
      expect(Preprocessor.normalizePixel(0), closeTo(-1.0, 1e-12));
      expect(Preprocessor.normalizePixel(127.5), closeTo(0.0, 1e-12));
      expect(Preprocessor.normalizePixel(255), closeTo(1.0, 1e-12));
    });

    test('is NOT plain /255 (that would keep values in [0,1])', () {
      expect(Preprocessor.normalizePixel(0), isNot(closeTo(0.0, 1e-6)));
    });

    test('is NOT ImageNet mean/std', () {
      const imagenetRForZero = (0 / 255.0 - 0.485) / 0.229;
      expect(Preprocessor.normalizePixel(0),
          isNot(closeTo(imagenetRForZero, 1e-3)));
    });

    test('a constant all-127 image -> tensor ~ -0.00392 (near 0), not 0.498', () {
      // 127/255*2-1 = -0.00392... A plain /255 would give 0.498 instead.
      final image = img.Image(width: 500, height: 420, numChannels: 3);
      img.fill(image, color: img.ColorRgb8(127, 127, 127));
      final data = Preprocessor.imageToTensor(image);
      final expected = Preprocessor.normalizePixel(127);
      expect(expected, closeTo(-0.00392, 1e-4));
      for (final v in [data[nchw(0, 0, 0)], data[nchw(1, 200, 200)],
        data[nchw(2, 383, 383)]]) {
        expect(v, closeTo(expected, 1e-4));
      }
    });
  });

  group('channel order (RGB, not BGR) + tensor shape', () {
    test('output length and shape are [1,3,384,384]', () {
      final image = img.Image(width: 500, height: 460, numChannels: 3);
      img.fill(image, color: img.ColorRgb8(10, 20, 30));
      final data = Preprocessor.imageToTensor(image);
      expect(data.length, Preprocessor.tensorLength);
      expect(Preprocessor.tensorShape, [1, 3, 384, 384]);
    });

    test('R,G,B constants land in channels 0,1,2 in that order', () {
      // Distinct per-channel constants prove RGB (not BGR) and per-channel norm.
      final image = img.Image(width: 500, height: 500, numChannels: 3);
      img.fill(image, color: img.ColorRgb8(255, 127, 0));
      final data = Preprocessor.imageToTensor(image);
      final expectedR = Preprocessor.normalizePixel(255); // +1.0
      final expectedG = Preprocessor.normalizePixel(127); // ~ -0.003
      final expectedB = Preprocessor.normalizePixel(0); // -1.0
      for (final (y, x) in [(0, 0), (100, 50), (383, 383), (192, 192)]) {
        expect(data[nchw(0, y, x)], closeTo(expectedR, 1e-4));
        expect(data[nchw(1, y, x)], closeTo(expectedG, 1e-4));
        expect(data[nchw(2, y, x)], closeTo(expectedB, 1e-4));
      }
    });

    test('all normalized values stay within [-1, 1]', () {
      final image = img.Image(width: 640, height: 400, numChannels: 3);
      img.fill(image, color: img.ColorRgb8(200, 50, 250));
      final data = Preprocessor.imageToTensor(image);
      for (final v in data) {
        expect(v, greaterThanOrEqualTo(-1.0 - 1e-6));
        expect(v, lessThanOrEqualTo(1.0 + 1e-6));
      }
    });
  });

  group('resize shortest-edge -> 384 then center-crop 384 (NOT a squash)', () {
    test('resize preserves aspect ratio (shortest edge -> 384, floor)', () {
      expect(Preprocessor.resizedDimensions(768, 384), (768, 384));
      expect(Preprocessor.resizedDimensions(1000, 500), (768, 384));
      expect(Preprocessor.resizedDimensions(500, 1000), (384, 768));
      expect(Preprocessor.resizedDimensions(640, 640), (384, 384));
    });

    test('a non-square image is never squashed to a 384x384 resize', () {
      final (w, h) = Preprocessor.resizedDimensions(800, 400);
      expect(w == h, isFalse);
      expect(w != Preprocessor.inputSize || h != Preprocessor.inputSize, isTrue);
    });

    test('center-crop origin takes the middle 384x384 (floor)', () {
      expect(Preprocessor.cropOrigin(768, 384), (192, 0));
      expect(Preprocessor.cropOrigin(384, 384), (0, 0));
      expect(Preprocessor.cropOrigin(384, 768), (0, 192));
    });
  });

  group('cubic kernel (Keys a=-0.5) shape', () {
    test('kernel(0)=1, kernel(1)=0, kernel(2)=0, symmetric, negative lobe', () {
      expect(Preprocessor.cubicKernel(0), closeTo(1.0, 1e-12));
      expect(Preprocessor.cubicKernel(1), closeTo(0.0, 1e-12));
      expect(Preprocessor.cubicKernel(2), closeTo(0.0, 1e-12));
      expect(Preprocessor.cubicKernel(-0.5),
          closeTo(Preprocessor.cubicKernel(0.5), 1e-12));
      // Between 1 and 2 the Keys kernel is negative (the anti-alias lobe).
      expect(Preprocessor.cubicKernel(1.5), lessThan(0.0));
      expect(Preprocessor.cubicKernel(3), 0.0);
    });
  });
}
