import 'package:dentalxraydetect/services/preprocessor.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('letterbox @ 640', () {
    test('target is 640, not 928/640-confusion', () {
      expect(computeLetterbox(640, 640).target, 640);
    });

    test('test_letterbox_inverse_roundtrip returns a known box to origin', () {
      // A real panoramic radiograph resolution (val_28 is 2872x1504).
      final LetterboxParams p = computeLetterbox(2872, 1504);
      // Width-limited: scale = 640/2872.
      expect(p.scale, closeTo(640 / 2872, 1e-9));
      expect(p.scaledW, 640);
      expect(p.padX, closeTo(0, 0.5));
      expect(p.padY, greaterThan(0));

      // Forward-map a known source point into 640 model space, then invert.
      const double sx = 1436, sy = 752; // image center
      final double mx = sx * p.scale + p.padX;
      final double my = sy * p.scale + p.padY;
      expect(p.modelToSrcX(mx), closeTo(sx, 1e-6));
      expect(p.modelToSrcY(my), closeTo(sy, 1e-6));

      // A corner too (0,0) -> pad origin -> back to (0,0).
      expect(p.modelToSrcX(0 * p.scale + p.padX), closeTo(0, 1e-6));
      expect(p.modelToSrcY(0 * p.scale + p.padY), closeTo(0, 1e-6));
    });

    test('portrait source pads horizontally', () {
      final LetterboxParams p = computeLetterbox(1504, 2872);
      expect(p.scale, closeTo(640 / 2872, 1e-9));
      expect(p.scaledH, 640);
      expect(p.padY, closeTo(0, 0.5));
      expect(p.padX, greaterThan(0));
    });

    test('square source has unit scale and zero pad', () {
      final LetterboxParams p = computeLetterbox(640, 640);
      expect(p.scale, closeTo(1.0, 1e-9));
      expect(p.padX, 0);
      expect(p.padY, 0);
    });
  });
}
