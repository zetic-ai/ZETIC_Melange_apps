import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:shelfscanyolo/models/detection.dart';
import 'package:shelfscanyolo/services/display_fit.dart';
import 'package:shelfscanyolo/services/letterbox.dart';

void main() {
  group('letterbox params match validate_demo.py', () {
    test('portrait 3120x4160 -> scale, pad, resized', () {
      final t = LetterboxTransform.compute(
        originalWidth: 3120,
        originalHeight: 4160,
      );
      expect(t.scale, closeTo(640 / 4160, 1e-12));
      expect(t.resizedWidth, 480); // round(3120 * 640/4160)
      expect(t.resizedHeight, 640);
      expect(t.padX, 80); // (640-480)//2
      expect(t.padY, 0);
    });

    test('landscape 2592x1936 -> pads vertically', () {
      final t = LetterboxTransform.compute(
        originalWidth: 2592,
        originalHeight: 1936,
      );
      expect(t.scale, closeTo(640 / 2592, 1e-12));
      expect(t.resizedWidth, 640);
      expect(t.padX, 0);
      expect(t.padY, greaterThan(0));
      // Pads are symmetric via floor division.
      expect(t.padY, (640 - t.resizedHeight) ~/ 2);
    });
  });

  group('letterbox inverse round-trip', () {
    test('original -> letterbox -> original returns the same box', () {
      final t = LetterboxTransform.compute(
        originalWidth: 3120,
        originalHeight: 4160,
      );
      const b = BBox(500, 800, 700, 1100);
      // forward
      final lx1 = t.toLetterboxX(b.x1);
      final ly1 = t.toLetterboxY(b.y1);
      final lx2 = t.toLetterboxX(b.x2);
      final ly2 = t.toLetterboxY(b.y2);
      // forward coords must land inside the 640 canvas
      for (final v in [lx1, ly1, lx2, ly2]) {
        expect(v, inInclusiveRange(0.0, 640.0));
      }
      // inverse
      final back = t.letterboxToOriginal(BBox(lx1, ly1, lx2, ly2));
      expect(back.x1, closeTo(b.x1, 1e-6));
      expect(back.y1, closeTo(b.y1, 1e-6));
      expect(back.x2, closeTo(b.x2, 1e-6));
      expect(back.y2, closeTo(b.y2, 1e-6));
    });

    test('full image box maps to full letterbox content region', () {
      final t = LetterboxTransform.compute(
        originalWidth: 3120,
        originalHeight: 4160,
      );
      // The whole original image occupies the letterbox content rect.
      expect(t.toLetterboxX(0), closeTo(80, 1e-6));
      expect(t.toLetterboxX(3120), closeTo(560, 1e-6)); // 80 + 480
      expect(t.toLetterboxY(0), closeTo(0, 1e-6));
      expect(t.toLetterboxY(4160), closeTo(640, 1e-6));
    });
  });

  group('displayed-image fit (BoxFit.contain) round-trip', () {
    test('letterbox bars and mapping for a portrait image in a square widget',
        () {
      final fit = DisplayFit.contain(
        imageWidth: 3120,
        imageHeight: 4160,
        widgetWidth: 400,
        widgetHeight: 400,
      );
      final scale = math.min(400 / 3120, 400 / 4160);
      expect(fit.scale, closeTo(scale, 1e-12));
      // Image displayed height fills 400; width has bars of 50px each side.
      expect(fit.dx, closeTo(50, 1e-6));
      expect(fit.dy, closeTo(0, 1e-6));
      expect(fit.mapX(0), closeTo(50, 1e-6));
      expect(fit.mapX(3120), closeTo(350, 1e-6));
      expect(fit.mapY(0), closeTo(0, 1e-6));
      expect(fit.mapY(4160), closeTo(400, 1e-6));
    });

    test('end-to-end: 640 letterbox box -> original px -> screen px', () {
      final t = LetterboxTransform.compute(
        originalWidth: 3120,
        originalHeight: 4160,
      );
      final fit = DisplayFit.contain(
        imageWidth: 3120,
        imageHeight: 4160,
        widgetWidth: 400,
        widgetHeight: 400,
      );
      // A box covering the full letterbox content region (80..560, 0..640).
      final orig = t.letterboxToOriginal(const BBox(80, 0, 560, 640));
      expect(orig.x1, closeTo(0, 1e-6));
      expect(orig.y1, closeTo(0, 1e-6));
      expect(orig.x2, closeTo(3120, 1e-6));
      expect(orig.y2, closeTo(4160, 1e-6));
      final screen = fit.mapBox(orig);
      expect(screen[0], closeTo(50, 1e-6)); // left bar
      expect(screen[1], closeTo(0, 1e-6));
      expect(screen[2], closeTo(350, 1e-6)); // right edge of image
      expect(screen[3], closeTo(400, 1e-6));
    });
  });
}
