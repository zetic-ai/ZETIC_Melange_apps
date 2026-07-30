import 'dart:ui';

import 'package:aerialdetect/services/preprocessor.dart';
import 'package:aerialdetect/widgets/coordinate_mapping.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('still-image letterbox @ 928', () {
    test('landscape aerial still letterboxes at 928 and inverts exactly', () {
      // A typical (downscaled) aerial still: wider than tall.
      final LetterboxParams p = computeLetterbox(2048, 1152);
      expect(p.target, 928, reason: 'still path must letterbox at 928');
      // scale = 928 / 2048 (width-limited).
      expect(p.scale, closeTo(928 / 2048, 1e-9));
      expect(p.scaledW, 928);
      expect(p.padX, closeTo(0, 0.5));
      expect(p.padY, greaterThan(0));

      // Forward-map a source point into 928 model space, then invert back.
      const double sx = 1024, sy = 576;
      final double mx = sx * p.scale + p.padX;
      final double my = sy * p.scale + p.padY;
      expect(p.modelToSrcX(mx), closeTo(sx, 1e-6));
      expect(p.modelToSrcY(my), closeTo(sy, 1e-6));
    });
  });

  group('BoxFit.contain overlay mapping', () {
    test('contain fit centers a landscape image with vertical letterbox', () {
      const Size image = Size(2048, 1152);
      const Size canvas = Size(400, 800); // taller canvas → pillar/letterbox top-bottom
      final ContainFit f = computeContainFit(image, canvas);
      // Width-limited: scale = 400/2048.
      expect(f.scale, closeTo(400 / 2048, 1e-9));
      expect(f.dx, closeTo(0, 1e-6));
      expect(f.dy, greaterThan(0));
      // Whole image fits inside the canvas.
      expect(f.destRect.left, greaterThanOrEqualTo(-1e-6));
      expect(f.destRect.right, lessThanOrEqualTo(canvas.width + 1e-6));
      expect(f.destRect.top, greaterThanOrEqualTo(-1e-6));
      expect(f.destRect.bottom, lessThanOrEqualTo(canvas.height + 1e-6));
    });

    test('mapContainRect round-trips through unmapContainRect', () {
      const Size image = Size(2048, 1152);
      const Size canvas = Size(390, 844);
      const Rect src = Rect.fromLTRB(100, 80, 900, 700);
      final Rect mapped = mapContainRect(src, image, canvas);
      final Rect back = unmapContainRect(mapped, image, canvas);
      expect(back.left, closeTo(src.left, 1e-6));
      expect(back.top, closeTo(src.top, 1e-6));
      expect(back.right, closeTo(src.right, 1e-6));
      expect(back.bottom, closeTo(src.bottom, 1e-6));
    });

    test('a full-frame box maps onto the drawn image dest rect', () {
      const Size image = Size(1600, 1200);
      const Size canvas = Size(500, 500);
      final ContainFit f = computeContainFit(image, canvas);
      final Rect full = mapContainRect(
        const Rect.fromLTRB(0, 0, 1600, 1200),
        image,
        canvas,
      );
      // The box covering the whole source must coincide with where the image
      // is painted — the single shared transform guarantees no drift.
      expect(full.left, closeTo(f.destRect.left, 1e-6));
      expect(full.top, closeTo(f.destRect.top, 1e-6));
      expect(full.right, closeTo(f.destRect.right, 1e-6));
      expect(full.bottom, closeTo(f.destRect.bottom, 1e-6));
    });
  });
}
