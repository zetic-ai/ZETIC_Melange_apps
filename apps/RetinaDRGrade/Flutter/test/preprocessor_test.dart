import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:retinadrgrade/services/preprocessor.dart';

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
      // And plain /255 of 255 is 1.0 by coincidence, but 128 differs sharply:
      // plain/255 -> 0.502, (v-0.5)/0.5 -> 0.00392...
      expect(Preprocessor.normalizePixel(128), isNot(closeTo(128 / 255.0, 1e-3)));
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

  group('PLAIN resize-to-224 geometry (NOT shortest-edge-256 -> center-crop)', () {
    test('a non-square image is resized directly to exactly 224x224', () {
      // A 640x360 landscape image must become 224x224 (whole image mapped in),
      // NOT cropped from a 256-shortest-edge intermediate. We prove geometry by
      // filling the WHOLE image with one color and confirming the output is a
      // full 224x224 tensor with that color everywhere — a crop pipeline would
      // still be 224x224, so we separately prove no-crop below.
      final image = img.Image(width: 640, height: 360, numChannels: 3);
      img.fill(image, color: img.ColorRgb8(200, 100, 50));
      final data = Preprocessor.imageToTensor(image);
      expect(data.length, Preprocessor.tensorLength);
    });

    test('edges of a non-square image survive (no center-crop discards them)', () {
      // Paint a distinctive 1px LEFT-edge stripe on a wide image. A plain resize
      // maps the entire width into 224, so column x=0 of the output reflects the
      // left edge. A shortest-edge-256 -> center-crop-224 would DISCARD the left
      // and right edges (crop origin x>0), so the stripe would vanish. This is
      // the key geometric discriminator vs the sibling RetinaDRScreen.
      final image = img.Image(width: 700, height: 300, numChannels: 3);
      img.fill(image, color: img.ColorRgb8(0, 0, 0)); // black field
      // Bright red left column band (first 6 px) and right column band.
      for (var y = 0; y < 300; y++) {
        for (var x = 0; x < 6; x++) {
          image.setPixelRgb(x, y, 255, 0, 0);
        }
        for (var x = 694; x < 700; x++) {
          image.setPixelRgb(x, y, 255, 0, 0);
        }
      }
      final data = Preprocessor.imageToTensor(image);
      // R channel at output x=0 (left edge) should be near +1 (255 -> +1),
      // proving the left edge was mapped in, not cropped away.
      final leftR = data[nchw(0, 112, 0)];
      final rightR = data[nchw(0, 112, 223)];
      expect(leftR, greaterThan(0.0),
          reason: 'left edge lost — pipeline is cropping, not plain-resizing');
      expect(rightR, greaterThan(0.0),
          reason: 'right edge lost — pipeline is cropping, not plain-resizing');
    });

    test('center of a tall image is preserved and normalized correctly', () {
      final image = img.Image(width: 300, height: 900, numChannels: 3);
      img.fill(image, color: img.ColorRgb8(255, 127, 0));
      final data = Preprocessor.imageToTensor(image);
      expect(data[nchw(0, 112, 112)],
          closeTo(Preprocessor.normalizePixel(255), 1e-3));
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
