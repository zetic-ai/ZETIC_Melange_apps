import 'dart:typed_data';
import 'dart:ui' show Rect;

import 'package:flutter_test/flutter_test.dart';
import 'package:snapseek/services/preprocessor.dart';

void main() {
  group('crop-space mapping (original-frame px, not letterbox space)', () {
    test('box maps to original-frame pixels with margin, clamped to bounds', () {
      // Frame 1000x500; box normalized (0.2,0.1,0.6,0.5) → px (200,50,600,250).
      final req = EmbedRequest(
        width: 1000,
        height: 500,
        rgb: Uint8List(1000 * 500 * 3),
        box: const Rect.fromLTRB(0.2, 0.1, 0.6, 0.5),
        margin: 0.06,
      );
      final crop = cropRectFor(req);
      // margin: 0.06 * 400px width = 24; 0.06 * 200px height = 12.
      expect(crop.left, closeTo(176, 1e-6));
      expect(crop.top, closeTo(38, 1e-6));
      expect(crop.right, closeTo(624, 1e-6));
      expect(crop.bottom, closeTo(262, 1e-6));
    });

    test('margin expansion clamps at the frame edges', () {
      final req = EmbedRequest(
        width: 400,
        height: 400,
        rgb: Uint8List(400 * 400 * 3),
        box: const Rect.fromLTRB(0.0, 0.0, 1.0, 1.0),
        margin: 0.10,
      );
      final crop = cropRectFor(req);
      expect(crop, const Rect.fromLTRB(0, 0, 400, 400));
    });

    test('null box → centered square crop (fallback)', () {
      final req = EmbedRequest(
        width: 1000,
        height: 400,
        rgb: Uint8List(1000 * 400 * 3),
        box: null,
      );
      final crop = cropRectFor(req);
      // side = min(1000,400) = 400, centered on (500,200).
      expect(crop.left, closeTo(300, 1e-6));
      expect(crop.top, closeTo(0, 1e-6));
      expect(crop.right, closeTo(700, 1e-6));
      expect(crop.bottom, closeTo(400, 1e-6));
    });

    test('preprocessEmbed crops the RIGHT region (not letterbox space)', () {
      // 100x100 frame: left half red, right half blue. Box selects the right
      // half → the embed tensor's R channel must be ~0 and B channel ~1.
      const w = 100, h = 100;
      final rgb = Uint8List(w * h * 3);
      for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
          final i = (y * w + x) * 3;
          if (x < 50) {
            rgb[i] = 255; // red left
          } else {
            rgb[i + 2] = 255; // blue right
          }
        }
      }
      final bundle = preprocessEmbed(EmbedRequest(
        width: w,
        height: h,
        rgb: rgb,
        box: const Rect.fromLTRB(0.55, 0.1, 0.95, 0.9), // right (blue) region
        margin: 0.0,
      ));
      const area = kEmbedSize * kEmbedSize;
      // Sample a central pixel of the 256x256 output.
      const p = (kEmbedSize ~/ 2) * kEmbedSize + kEmbedSize ~/ 2;
      final r = bundle.input[p];
      final b = bundle.input[2 * area + p];
      expect(r, closeTo(0.0, 0.02));
      expect(b, closeTo(1.0, 0.02));
      expect(bundle.input.length, 3 * area);
    });
  });
}
