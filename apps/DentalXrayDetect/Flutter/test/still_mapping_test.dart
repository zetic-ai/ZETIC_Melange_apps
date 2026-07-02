import 'dart:ui';

import 'package:dentalxraydetect/services/preprocessor.dart';
import 'package:dentalxraydetect/widgets/coordinate_mapping.dart';
import 'package:flutter_test/flutter_test.dart';

/// The full box path is two composed transforms:
///   1. letterbox inverse (640 model space -> original-image pixel space), and
///   2. BoxFit.contain (original-image pixel space -> on-screen canvas).
/// Both must round-trip a known box exactly, or the overlay drifts off the
/// radiograph.
void main() {
  group('letterbox inverse -> original-image space @ 640', () {
    test('a landscape radiograph inverts a full box round-trip', () {
      final LetterboxParams p = computeLetterbox(2872, 1504);
      const double sx = 500, sy = 900;
      final double mx = sx * p.scale + p.padX;
      final double my = sy * p.scale + p.padY;
      expect(p.modelToSrcX(mx), closeTo(sx, 1e-6));
      expect(p.modelToSrcY(my), closeTo(sy, 1e-6));
    });
  });

  group('BoxFit.contain overlay mapping', () {
    test('contain fit centers a wide radiograph with vertical letterbox', () {
      const Size image = Size(2872, 1504);
      const Size canvas = Size(400, 800); // taller canvas -> top/bottom bars
      final ContainFit f = computeContainFit(image, canvas);
      // Width-limited: scale = 400/2872.
      expect(f.scale, closeTo(400 / 2872, 1e-9));
      expect(f.dx, closeTo(0, 1e-6));
      expect(f.dy, greaterThan(0));
      // Whole image fits inside the canvas.
      expect(f.destRect.left, greaterThanOrEqualTo(-1e-6));
      expect(f.destRect.right, lessThanOrEqualTo(canvas.width + 1e-6));
      expect(f.destRect.top, greaterThanOrEqualTo(-1e-6));
      expect(f.destRect.bottom, lessThanOrEqualTo(canvas.height + 1e-6));
    });

    test('mapContainRect round-trips through unmapContainRect', () {
      const Size image = Size(2872, 1504);
      const Size canvas = Size(390, 844);
      const Rect src = Rect.fromLTRB(120, 90, 1800, 1300);
      final Rect mapped = mapContainRect(src, image, canvas);
      final Rect back = unmapContainRect(mapped, image, canvas);
      expect(back.left, closeTo(src.left, 1e-6));
      expect(back.top, closeTo(src.top, 1e-6));
      expect(back.right, closeTo(src.right, 1e-6));
      expect(back.bottom, closeTo(src.bottom, 1e-6));
    });

    test('a full-frame box maps onto the drawn image dest rect', () {
      const Size image = Size(2872, 1504);
      const Size canvas = Size(500, 500);
      final ContainFit f = computeContainFit(image, canvas);
      final Rect full = mapContainRect(
        const Rect.fromLTRB(0, 0, 2872, 1504),
        image,
        canvas,
      );
      // The box covering the whole source coincides with where the image is
      // painted — the single shared transform guarantees no drift.
      expect(full.left, closeTo(f.destRect.left, 1e-6));
      expect(full.top, closeTo(f.destRect.top, 1e-6));
      expect(full.right, closeTo(f.destRect.right, 1e-6));
      expect(full.bottom, closeTo(f.destRect.bottom, 1e-6));
    });
  });
}
