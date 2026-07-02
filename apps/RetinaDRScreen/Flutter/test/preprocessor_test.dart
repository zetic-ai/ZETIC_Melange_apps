import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:retinadrscreen/services/preprocessor.dart';

/// Helper: index into the NCHW [1,3,224,224] flat buffer.
int nchw(int c, int y, int x) =>
    c * Preprocessor.inputSize * Preprocessor.inputSize +
    y * Preprocessor.inputSize +
    x;

void main() {
  group('(v-0.5)/0.5 normalization exactness', () {
    test('maps 0 -> -1, 127.5 -> 0, 255 -> +1', () {
      expect(Preprocessor.normalizePixel(0), closeTo(-1.0, 1e-12));
      expect(Preprocessor.normalizePixel(127.5), closeTo(0.0, 1e-12));
      expect(Preprocessor.normalizePixel(255), closeTo(1.0, 1e-12));
    });

    test('is NOT plain /255 (that would keep values in [0,1])', () {
      // Plain /255 of 0 is 0.0; the correct normalization of 0 is -1.0.
      expect(Preprocessor.normalizePixel(0), isNot(closeTo(0.0, 1e-6)));
    });

    test('is NOT ImageNet mean/std', () {
      // ImageNet R would be (0/255 - 0.485)/0.229 = -2.1179..., not -1.0.
      const imagenetRForZero = (0 / 255.0 - 0.485) / 0.229;
      expect(
        Preprocessor.normalizePixel(0),
        isNot(closeTo(imagenetRForZero, 1e-3)),
      );
    });
  });

  group('resize shortest-edge -> 256 then center-crop 224 (NOT a squash)', () {
    test('resize preserves aspect ratio (shortest edge -> 256)', () {
      // A squash-to-224 would return (224,224); the correct pipeline does not.
      expect(Preprocessor.resizedDimensions(512, 256), (512, 256));
      expect(Preprocessor.resizedDimensions(1000, 500), (512, 256));
      expect(Preprocessor.resizedDimensions(500, 1000), (256, 512));
      // Square: shortest edge scales to 256.
      expect(Preprocessor.resizedDimensions(640, 640), (256, 256));
    });

    test('a non-square image is never squashed to a 224x224 resize', () {
      final (w, h) = Preprocessor.resizedDimensions(800, 400);
      expect(w == h, isFalse);
      expect(w != Preprocessor.inputSize || h != Preprocessor.inputSize, isTrue);
    });

    test('center-crop origin takes the middle 224x224', () {
      expect(Preprocessor.cropOrigin(512, 256), (144, 16));
      expect(Preprocessor.cropOrigin(256, 256), (16, 16));
      expect(Preprocessor.cropOrigin(256, 512), (16, 144));
    });
  });

  group('channel order (RGB) + tensor shape', () {
    test('output length and shape are [1,3,224,224]', () {
      final image = img.Image(width: 300, height: 260, numChannels: 3);
      img.fill(image, color: img.ColorRgb8(10, 20, 30));
      final data = Preprocessor.imageToTensor(image);
      expect(data.length, Preprocessor.tensorLength);
      expect(Preprocessor.tensorShape, [1, 3, 224, 224]);
    });

    test('R,G,B constants land in channels 0,1,2 in that order', () {
      // Distinct per-channel constants prove RGB order (not BGR) and that the
      // normalization is applied per channel.
      final image = img.Image(width: 260, height: 260, numChannels: 3);
      img.fill(image, color: img.ColorRgb8(255, 127, 0));
      final data = Preprocessor.imageToTensor(image);

      final expectedR = Preprocessor.normalizePixel(255); // +1.0
      final expectedG = Preprocessor.normalizePixel(127); // ~ -0.003...
      final expectedB = Preprocessor.normalizePixel(0); // -1.0

      // Sample a few interior pixels.
      for (final (y, x) in [(0, 0), (100, 50), (223, 223), (112, 112)]) {
        expect(data[nchw(0, y, x)], closeTo(expectedR, 1e-4));
        expect(data[nchw(1, y, x)], closeTo(expectedG, 1e-4));
        expect(data[nchw(2, y, x)], closeTo(expectedB, 1e-4));
      }
    });

    test('all normalized values stay within [-1, 1]', () {
      final image = img.Image(width: 400, height: 300, numChannels: 3);
      img.fill(image, color: img.ColorRgb8(200, 50, 250));
      final data = Preprocessor.imageToTensor(image);
      for (final v in data) {
        expect(v, greaterThanOrEqualTo(-1.0 - 1e-6));
        expect(v, lessThanOrEqualTo(1.0 + 1e-6));
      }
    });
  });
}
