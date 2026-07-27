import 'dart:typed_data';
import 'dart:ui' show Rect;

import 'package:flutter_test/flutter_test.dart';
import 'package:snapseek/services/preprocessor.dart';

void main() {
  group('embed preprocessing — plain /255, NO ImageNet mean/std', () {
    const area = kEmbedSize * kEmbedSize;

    test('mid-gray 128 → 0.502 on all channels (no mean subtraction)', () {
      const w = 40, h = 40;
      final rgb = Uint8List(w * h * 3)..fillRange(0, w * h * 3, 128);
      final b = preprocessEmbed(EmbedRequest(
          width: w, height: h, rgb: rgb, box: null, margin: 0.0));
      // If ImageNet mean/std were (wrongly) applied, R would be ~ (0.502-0.485)/
      // 0.229 ≈ 0.074, not 0.502. Assert plain /255.
      for (final p in [0, area, 2 * area]) {
        expect(b.input[p + area ~/ 2], closeTo(128 / 255.0, 1e-4));
      }
    });

    test('pure red pixel → R=1, G=0, B=0 (channel order + /255)', () {
      const w = 20, h = 20;
      final rgb = Uint8List(w * h * 3);
      for (var i = 0; i < w * h; i++) {
        rgb[i * 3] = 255; // R only
      }
      final b = preprocessEmbed(EmbedRequest(
          width: w, height: h, rgb: rgb, box: null, margin: 0.0));
      const p = (kEmbedSize ~/ 2) * kEmbedSize + kEmbedSize ~/ 2;
      expect(b.input[p], closeTo(1.0, 1e-4));
      expect(b.input[area + p], closeTo(0.0, 1e-4));
      expect(b.input[2 * area + p], closeTo(0.0, 1e-4));
    });

    test('output is exactly [1,3,256,256] and within [0,1]', () {
      final rgb = Uint8List(64 * 64 * 3);
      for (var i = 0; i < rgb.length; i++) {
        rgb[i] = (i * 37) % 256;
      }
      final b = preprocessEmbed(EmbedRequest(
          width: 64,
          height: 64,
          rgb: rgb,
          box: const Rect.fromLTRB(0.1, 0.1, 0.9, 0.9)));
      expect(b.input.length, 3 * area);
      for (final v in b.input) {
        expect(v, inInclusiveRange(0.0, 1.0));
      }
    });
  });
}
