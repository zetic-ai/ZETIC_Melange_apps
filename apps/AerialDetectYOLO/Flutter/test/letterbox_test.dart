import 'package:aerialdetect/services/preprocessor.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('letterbox @ 928', () {
    test('test_letterbox_inverse_roundtrip_928 returns a known box to origin',
        () {
      // Portrait source; a hard-coded 640 would change scale/pad and fail.
      final LetterboxParams p = computeLetterbox(720, 1280);
      expect(p.target, 928, reason: 'must letterbox at 928, not 640');
      expect(p.scale, closeTo(0.725, 1e-3));
      expect(p.padX, closeTo(203, 0.5));
      expect(p.padY, closeTo(0, 0.5));

      // Forward-map a known source point into model space, then invert.
      const double sx = 360, sy = 640;
      final double mx = sx * p.scale + p.padX;
      final double my = sy * p.scale + p.padY;
      expect(p.modelToSrcX(mx), closeTo(sx, 1e-6));
      expect(p.modelToSrcY(my), closeTo(sy, 1e-6));
    });

    test('test_letterbox_scale_pad_computation_928 (landscape source)', () {
      final LetterboxParams p = computeLetterbox(1280, 720);
      expect(p.scale, closeTo(0.725, 1e-3));
      expect(p.scaledW, 928);
      expect(p.scaledH, 522);
      expect(p.padX, closeTo(0, 0.5));
      expect(p.padY, closeTo(203, 0.5));
    });

    test('square source has unit scale and zero pad', () {
      final LetterboxParams p = computeLetterbox(928, 928);
      expect(p.scale, closeTo(1.0, 1e-9));
      expect(p.padX, 0);
      expect(p.padY, 0);
    });
  });
}
