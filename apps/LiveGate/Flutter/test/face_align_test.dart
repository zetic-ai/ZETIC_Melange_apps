import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:livegate/models/geometry.dart';
import 'package:livegate/services/face_align.dart';
import 'package:livegate/services/frame.dart';

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
  group('T4 FACE channel order + range (BGR, [0,255], no normalization)', () {
    test('warp output keeps BGR order and [0,255] range', () {
      final img = solid(200, 200, 12, 34, 56); // B, G, R
      const lm = Landmarks5(
        leftEye: Point2(70, 90),
        rightEye: Point2(130, 90),
        nose: Point2(100, 120),
        mouthLeft: Point2(80, 150),
        mouthRight: Point2(120, 150),
      );
      final out = buildFaceInput(img, lm);
      const area = kFaceInputSize * kFaceInputSize;
      // Uniform image => every warped pixel is the same BGR triple, unscaled.
      expect(out[0], closeTo(12, 0.5));
      expect(out[area], closeTo(34, 0.5));
      expect(out[2 * area], closeTo(56, 0.5));
      // No ÷255 anywhere.
      var maxV = 0.0;
      for (final v in out) {
        if (v > maxV) maxV = v;
      }
      expect(maxV, closeTo(56, 0.5));
    });
  });

  group('T6 FACE 5-point alignment (similarity transform)', () {
    test('template -> template is the identity', () {
      final t = estimateSimilarity(kArcFaceTemplate, kArcFaceTemplate);
      for (final p in kArcFaceTemplate) {
        final q = t.apply(p);
        expect(q.x, closeTo(p.x, 1e-6));
        expect(q.y, closeTo(p.y, 1e-6));
      }
    });

    test('recovers a known similarity (scale + rotation + translation)', () {
      const s = 1.8;
      const theta = 0.20; // radians
      const tx = 30.0, ty = -12.0;
      final cosT = math.cos(theta), sinT = math.sin(theta);
      // Apply a true similarity to the template to make the "source" points.
      final src = [
        for (final p in kArcFaceTemplate)
          Point2(
            s * (cosT * p.x - sinT * p.y) + tx,
            s * (sinT * p.x + cosT * p.y) + ty,
          ),
      ];
      // Fit template -> src, then check it maps the 5 template points onto src.
      final t = estimateSimilarity(kArcFaceTemplate, src);
      for (var i = 0; i < src.length; i++) {
        final q = t.apply(kArcFaceTemplate[i]);
        expect(q.x, closeTo(src[i].x, 1e-4));
        expect(q.y, closeTo(src[i].y, 1e-4));
      }
    });
  });
}
