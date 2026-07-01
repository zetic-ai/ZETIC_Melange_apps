import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:vehicleplateyolo/services/fit.dart';
import 'package:vehicleplateyolo/services/still_image.dart';

/// Pure-logic tests for the still-photo path: the downscale sizing applied to a
/// picked photo, and the BoxFit.contain mapping the overlay draws boxes with.
/// Decode (device codec) and OCR accuracy are NOT exercised here.
void main() {
  group('scaledDimensions (still downscale sizing)', () {
    test('leaves small images untouched (never upscales)', () {
      expect(scaledDimensions(800, 600, 1920), (800, 600));
      expect(scaledDimensions(1920, 1080, 1920), (1920, 1080));
    });

    test('caps the longer side to maxSide, preserving aspect', () {
      // Landscape 4000x3000 -> long side 4000 -> scale 1920/4000 = 0.48.
      expect(scaledDimensions(4000, 3000, 1920), (1920, 1440));
      // Portrait 3000x4000 -> long side is height.
      expect(scaledDimensions(3000, 4000, 1920), (1440, 1920));
    });

    test('never produces a zero dimension', () {
      final (w, h) = scaledDimensions(10000, 1, 1920);
      expect(w, 1920);
      expect(h, greaterThanOrEqualTo(1));
    });

    test('handles degenerate input safely', () {
      expect(scaledDimensions(0, 0, 1920), (0, 0));
    });
  });

  group('FitMapping.contain (still overlay box mapping)', () {
    test('letterboxes a wide image inside a tall box (vertical bars)', () {
      // 200x100 image into a 200x200 box: scale 1 (min), centered vertically.
      final m = FitMapping.contain(200, 100, 200, 200);
      expect(m.scale, 1.0);
      expect(m.dx, 0.0);
      expect(m.dy, 50.0); // (200 - 100) / 2
      // A box corner at image (50,25) maps to (50, 75).
      expect(m.mapX(50), 50.0);
      expect(m.mapY(25), 75.0);
    });

    test('pillarboxes a tall image inside a wide box (horizontal bars)', () {
      // 100x200 image into a 200x200 box: scale 1 (min), centered horizontally.
      final m = FitMapping.contain(100, 200, 200, 200);
      expect(m.scale, 1.0);
      expect(m.dx, 50.0);
      expect(m.dy, 0.0);
    });

    test('scales down a large image to fit', () {
      // 400x200 into 200x200: scale 0.5, letterboxed vertically.
      final m = FitMapping.contain(400, 200, 200, 200);
      expect(m.scale, 0.5);
      expect(m.dx, 0.0);
      expect(m.dy, 50.0);
      expect(m.mapX(400), 200.0); // right edge lands at box right
    });
  });

  group('encodePlateCropPng (readable plate crop)', () {
    // A solid 100x100 BGRA buffer (blue=B channel high) to crop from.
    StillImage solid() {
      const w = 100, h = 100;
      final bgra = Uint8List(w * h * 4);
      for (var i = 0; i < w * h; i++) {
        bgra[i * 4] = 200; // B
        bgra[i * 4 + 1] = 100; // G
        bgra[i * 4 + 2] = 50; // R
        bgra[i * 4 + 3] = 255; // A
      }
      return StillImage(bgra: bgra, width: w, height: h);
    }

    test('upscales a small plate crop so its short side is readable', () {
      final png = encodePlateCropPng(solid(), 40, 40, 60, 60);
      expect(png, isNotNull);
      final decoded = img.decodePng(png!);
      expect(decoded, isNotNull);
      // 20x20 box (square) upscaled: short side >= kPlateCropMinSide.
      final short = decoded!.width < decoded.height
          ? decoded.width
          : decoded.height;
      expect(short, greaterThanOrEqualTo(kPlateCropMinSide));
    });

    test('preserves BGRA -> RGB channel order in the crop', () {
      final png = encodePlateCropPng(solid(), 40, 40, 60, 60);
      final decoded = img.decodePng(png!)!;
      final p = decoded.getPixel(0, 0);
      // Source B=200 -> R, G=100, R=50 -> B.
      expect(p.r, 50);
      expect(p.g, 100);
      expect(p.b, 200);
    });

    test('returns null for a degenerate region', () {
      expect(encodePlateCropPng(solid(), 60, 60, 40, 40), isNull);
    });
  });

  group('FitMapping.cover (live overlay parity)', () {
    test('fills the box and centers the overflow', () {
      // 200x100 into 200x200 cover: scale 2 (max), overflow centered on X.
      final m = FitMapping.cover(200, 100, 200, 200);
      expect(m.scale, 2.0);
      expect(m.dx, -100.0); // (200 - 400) / 2
      expect(m.dy, 0.0);
    });
  });
}
