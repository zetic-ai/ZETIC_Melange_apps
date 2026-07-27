import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:livegate/models/geometry.dart';
import 'package:livegate/services/frame.dart';
import 'package:livegate/services/preprocessor.dart';

BgrImage solid(int w, int h, int b, int g, int r) {
  final bytes = Uint8List(w * h * 3);
  for (var i = 0; i < w * h; i++) {
    bytes[i * 3] = b;
    bytes[i * 3 + 1] = g;
    bytes[i * 3 + 2] = r;
  }
  return BgrImage(bgr: bytes, width: w, height: h);
}

void main() {
  group('T1 PAD input range / channel order (÷255-saturation guard)', () {
    test('keeps BGR order and [0,255] range — never divides by 255', () {
      final img = solid(80, 80, 200, 100, 50); // B=200, G=100, R=50
      final crop = CropBox(0, 0, 79, 79);
      final out = buildPadInput(img, crop);
      const area = kPadInputSize * kPadInputSize;

      // Channel 0 = B, 1 = G, 2 = R, all in [0,255], NOT normalized to [0,1].
      expect(out[0], closeTo(200, 0.5));
      expect(out[area], closeTo(100, 0.5));
      expect(out[2 * area], closeTo(50, 0.5));

      var maxV = 0.0;
      for (final v in out) {
        if (v > maxV) maxV = v;
      }
      // The bug (÷255) would cap this at ~0.78; correct path keeps it at 200.
      expect(maxV, greaterThan(1.0),
          reason: 'range must be [0,255]; a value >1 proves no ÷255 saturation');
      expect(maxV, closeTo(200, 0.5));
    });
  });

  group('T3 PAD 2.7x crop clip-SHIFT geometry', () {
    test('centered face uses the full 2.7x margin', () {
      final crop = computeCropBox(
          1000, 1000, const FaceBox(x: 480, y: 480, w: 40, h: 40));
      // scale capped at 2.7; nw = nh = 108, centered on (500,500).
      expect(crop.width, closeTo(108, 1e-6));
      expect(crop.height, closeTo(108, 1e-6));
      expect((crop.x1 + crop.x2) / 2, closeTo(500, 1e-6));
    });

    test('edge face SHIFTS in-bounds preserving box size (not one-edge clamp)',
        () {
      // Face hard against the left edge.
      final crop = computeCropBox(
          200, 200, const FaceBox(x: 2, y: 80, w: 20, h: 40));
      // scale = min(199/40, 199/20, 2.7) = 2.7 -> nw=54, nh=108.
      expect(crop.x1, 0, reason: 'left edge shifted to 0');
      expect(crop.width, closeTo(54, 1e-6),
          reason: 'width preserved by SHIFT, not shrunk by a clamp');
      // Still fully inside the frame on every edge.
      expect(crop.x1 >= 0 && crop.y1 >= 0, isTrue);
      expect(crop.x2 <= 199 && crop.y2 <= 199, isTrue);
    });

    test('a huge face caps the scale at frame extent', () {
      final crop = computeCropBox(
          200, 200, const FaceBox(x: 20, y: 20, w: 150, h: 150));
      // (H-1)/bh = 199/150 = 1.327 < 2.7 -> scale capped.
      final expectedScale = 199 / 150;
      expect(crop.width, closeTo(150 * expectedScale, 1e-6));
    });
  });

  group('T10 golden fidelity (Dart == Python reference numerics)', () {
    test('buildPadInput reproduces the Python-generated golden NCHW', () {
      final file = File('test/fixtures/pad_golden.json');
      final json = jsonDecode(file.readAsStringSync()) as Map<String, dynamic>;

      final w = json['srcWidth'] as int;
      final h = json['srcHeight'] as int;
      final src = Uint8List.fromList(
          (json['srcBgr'] as List).map((e) => e as int).toList());
      final img = BgrImage(bgr: src, width: w, height: h);
      final c = json['crop'] as Map<String, dynamic>;
      final crop = CropBox((c['x1'] as num).toDouble(), (c['y1'] as num).toDouble(),
          (c['x2'] as num).toDouble(), (c['y2'] as num).toDouble());
      final expected = (json['expectedNchw'] as List)
          .map((e) => (e as num).toDouble())
          .toList();

      final out = buildPadInput(img, crop);
      expect(out.length, expected.length);
      for (var i = 0; i < expected.length; i++) {
        expect(out[i], closeTo(expected[i], 1e-6),
            reason: 'mismatch at index $i');
      }
    });
  });
}
