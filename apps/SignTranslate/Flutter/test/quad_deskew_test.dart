import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:signtranslate/config.dart';
import 'package:signtranslate/models/text_region.dart';
import 'package:signtranslate/services/detector_preprocessor.dart' show BgrFrame;
import 'package:signtranslate/services/quad_deskew.dart';

/// A synthetic "text-like" pattern: value depends on position so warps are
/// detectable (vertical bars of varying intensity + a horizontal gradient).
int patternValue(double u, double v) {
  final bar = ((u / 8).floor() % 2) * 160;
  final grad = (v * 2).round().clamp(0, 90);
  return (bar + grad).clamp(0, 255);
}

void main() {
  group('homography correctness', () {
    test('maps the dest rect corners exactly onto the quad corners', () {
      final quad = Quad(
        const Offset(120, 80),
        const Offset(300, 95),
        const Offset(290, 160),
        const Offset(115, 150),
      );
      const w = 180.0, h = 70.0;
      final hm = computeHomography(quad, w, h);

      expect((applyHomography(hm, 0, 0) - quad.tl).distance, lessThan(1e-6));
      expect((applyHomography(hm, w, 0) - quad.tr).distance, lessThan(1e-6));
      expect((applyHomography(hm, w, h) - quad.br).distance, lessThan(1e-6));
      expect((applyHomography(hm, 0, h) - quad.bl).distance, lessThan(1e-6));
    });

    test('interior points map projectively (midpoint of a true perspective '
        'quad is NOT the affine midpoint)', () {
      final quad = Quad(
        const Offset(0, 0),
        const Offset(100, 0),
        const Offset(80, 60),
        const Offset(10, 55),
      );
      final hm = computeHomography(quad, 100, 50);
      final mid = applyHomography(hm, 50, 25);
      // Interior and inside the hull.
      expect(mid.dx, inInclusiveRange(10.0, 100.0));
      expect(mid.dy, inInclusiveRange(0.0, 60.0));
    });

    test('degenerate (collinear) quads throw instead of dividing by ~0', () {
      final degenerate = Quad(
        const Offset(0, 0),
        const Offset(10, 0),
        const Offset(20, 0),
        const Offset(30, 0),
      );
      expect(() => computeHomography(degenerate, 10, 10), throwsStateError);
    });
  });

  group('deskew equivalence (the model-free proxy for "same string")', () {
    test('a perspective-tilted quad deskews back to the upright pattern', () {
      // 1. Ground truth: the upright pattern itself, 120x40.
      const cw = 120, ch = 40;

      // 2. Render a frame where that pattern has been perspective-warped
      //    into a tilted quad, by inverse-mapping every frame pixel.
      const fw = 400, fh = 300;
      final quad = Quad(
        const Offset(80, 60),
        const Offset(310, 90),
        const Offset(300, 175),
        const Offset(70, 130),
      );
      // Homography dest(rect uv) -> src(frame xy); invert per frame pixel by
      // using the forward map from rect to quad and scanning rect space.
      final hm = computeHomography(quad, cw.toDouble(), ch.toDouble());
      final frame = Uint8List(fw * fh * 3); // black background
      // Supersample rect space so the quad area in the frame is covered.
      for (var vy = 0; vy < ch * 8; vy++) {
        for (var vx = 0; vx < cw * 8; vx++) {
          final u = vx / 8.0, v = vy / 8.0;
          final p = applyHomography(hm, u, v);
          final x = p.dx.round(), y = p.dy.round();
          if (x < 0 || x >= fw || y < 0 || y >= fh) continue;
          final val = patternValue(u, v);
          final i = (y * fw + x) * 3;
          frame[i] = val;
          frame[i + 1] = val;
          frame[i + 2] = val;
        }
      }

      // 3. Deskew the quad out of the frame.
      final crop = deskewQuad(BgrFrame(fw, fh, frame), quad);

      // Not rotated (wide quad), sized from the quad edge lengths.
      expect(crop.width, greaterThan(crop.height));

      // 4. Compare the crop against the expected upright pattern at the
      //    crop's own scale: mean absolute error must be small.
      var err = 0.0;
      var n = 0;
      // Skip a 3px margin (bilinear edge effects + rounding).
      for (var y = 3; y < crop.height - 3; y++) {
        for (var x = 3; x < crop.width - 3; x++) {
          final u = (x + 0.5) * cw / crop.width;
          final v = (y + 0.5) * ch / crop.height;
          // Skip bar boundaries where a half-pixel shift flips the value.
          final uMod = (u / 8) - (u / 8).floorToDouble();
          if (uMod < 0.15 || uMod > 0.85) continue;
          final expected = patternValue(u, v);
          err += (crop.bgr[(y * crop.width + x) * 3] - expected).abs();
          n++;
        }
      }
      expect(n, greaterThan(1000));
      expect(err / n, lessThan(20),
          reason: 'deskewed crop must reproduce the upright pattern');
    });
  });

  group('vertical-crop rotate-90 rule (PaddleOCR parity, ruling #2)', () {
    test('a tall quad (h >= 1.5w) is rotated so the crop is wide', () {
      const fw = 200, fh = 300;
      final frame = Uint8List(fw * fh * 3);
      // Distinct corner marker: bright pixel near the quad's top-left.
      final quad = Quad(
        const Offset(50, 40),
        const Offset(110, 40), // width 60
        const Offset(110, 220), // height 180 = 3x width
        const Offset(50, 220),
      );
      frame[(45 * fw + 55) * 3] = 255; // near tl in frame space

      final crop = deskewQuad(BgrFrame(fw, fh, frame), quad);
      // 60x180 -> rotated CCW -> 180x60.
      expect(crop.width, closeTo(180, 2));
      expect(crop.height, closeTo(60, 2));
      expect(kVerticalCropRatio, 1.5);
    });

    test('a quad just below the ratio is NOT rotated', () {
      const fw = 200, fh = 300;
      final frame = Uint8List(fw * fh * 3);
      final quad = Quad(
        const Offset(50, 40),
        const Offset(110, 40), // width 60
        const Offset(110, 125), // height 85 < 90 = 1.5*60
        const Offset(50, 125),
      );
      final crop = deskewQuad(BgrFrame(fw, fh, frame), quad);
      expect(crop.width, closeTo(60, 2));
      expect(crop.height, closeTo(85, 2));
    });

    test('rotateCrop90Ccw moves pixels like np.rot90', () {
      // 3x2 crop with a marked pixel at (x=2, y=0).
      final bytes = Uint8List(3 * 2 * 3);
      bytes[(0 * 3 + 2) * 3] = 255;
      final rotated = rotateCrop90Ccw(BgrCrop(3, 2, bytes));
      expect(rotated.width, 2);
      expect(rotated.height, 3);
      // np.rot90: (x=2,y=0) -> row (w-1-x)=0, col y=0 -> (x=0,y=0).
      expect(rotated.bgr[0], 255);
    });
  });

  test('deskew of an axis-aligned quad equals a plain crop (sanity)', () {
    const fw = 100, fh = 100;
    final frame = Uint8List(fw * fh * 3);
    for (var y = 0; y < fh; y++) {
      for (var x = 0; x < fw; x++) {
        frame[(y * fw + x) * 3] = math.min(255, x * 2 + y);
      }
    }
    final quad = Quad(
      const Offset(10, 20),
      const Offset(70, 20),
      const Offset(70, 50),
      const Offset(10, 50),
    );
    final crop = deskewQuad(BgrFrame(fw, fh, frame), quad);
    expect(crop.width, 60);
    expect(crop.height, 30);
    // Sample center: frame pixel (40+0.5?, 35) ~ value x*2+y at (40,35).
    final centerVal = crop.bgr[(15 * 60 + 30) * 3];
    expect(centerVal, closeTo(40 * 2 + 35, 3));
  });
}
