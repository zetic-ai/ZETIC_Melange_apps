import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Offset, Rect;

import 'package:flutter_test/flutter_test.dart';
import 'package:signtranslate/config.dart';
import 'package:signtranslate/models/text_region.dart';
import 'package:signtranslate/services/db_postprocessor.dart';
import 'package:signtranslate/services/detector_preprocessor.dart';

const int size = kDetInputSize;

/// Identity geometry: frame space == 736 model space (no pad, scale 1).
final identityGeo = computeLetterboxGeometry(size, size);

Float32List emptyMap() => Float32List(size * size);

void fillRect(Float32List map, Rect r, double value) {
  for (var y = r.top.toInt(); y < r.bottom.toInt(); y++) {
    for (var x = r.left.toInt(); x < r.right.toInt(); x++) {
      map[y * size + x] = value;
    }
  }
}

void main() {
  test('DB parameters are SPEC-exact: 0.3 / 0.6 / 1.5', () {
    expect(kDbProbThreshold, 0.3);
    expect(kDbBoxThreshold, 0.6);
    expect(kDbUnclipRatio, 1.5);
  });

  group('no extra sigmoid (baked into the ONNX) — discriminator', () {
    test('a uniform 0.25 region yields ZERO boxes; sigmoid(0.25)=0.56 '
        'would wrongly binarize it as text', () {
      final map = emptyMap();
      fillRect(map, const Rect.fromLTWH(100, 100, 80, 30), 0.25);
      final result = dbPostProcess(map, identityGeo);
      expect(result.quads, isEmpty);
    });

    test('heatmap stats pass through raw (HUD anomaly check)', () {
      final map = emptyMap();
      fillRect(map, const Rect.fromLTWH(0, 0, 10, 10), 0.8);
      final result = dbPostProcess(map, identityGeo);
      expect(result.mapMax, closeTo(0.8, 1e-6));
      expect(result.mapMin, 0.0);
      expect(result.mapMean, closeTo(0.8 * 100 / (size * size), 1e-9));
    });
  });

  group('threshold boundaries', () {
    test('binarization at 0.3: 0.299 dropped, 0.301 kept', () {
      final below = emptyMap();
      fillRect(below, const Rect.fromLTWH(50, 50, 60, 20), 0.299);
      expect(dbPostProcess(below, identityGeo).quads, isEmpty);

      // 0.301 binarizes, but must also pass box_thresh -> use a mean above
      // 0.6 except the boundary probe: full region at 0.9 with ONE extra
      // pixel at 0.301 attached -> still one region (the pixel joined it).
      final above = emptyMap();
      fillRect(above, const Rect.fromLTWH(50, 50, 60, 20), 0.9);
      above[49 * size + 55] = 0.301; // touches the region from above
      final result = dbPostProcess(above, identityGeo);
      expect(result.quads, hasLength(1));
      // And a pure just-above-prob-thresh region fails box_thresh instead:
      final probOnly = emptyMap();
      fillRect(probOnly, const Rect.fromLTWH(50, 50, 60, 20), 0.301);
      expect(dbPostProcess(probOnly, identityGeo).quads, isEmpty);
    });

    test('box_thresh 0.6 on the region MEAN: 0.59 dropped, 0.61 kept', () {
      final low = emptyMap();
      fillRect(low, const Rect.fromLTWH(200, 200, 50, 18), 0.59);
      expect(dbPostProcess(low, identityGeo).quads, isEmpty);

      final high = emptyMap();
      fillRect(high, const Rect.fromLTWH(200, 200, 50, 18), 0.61);
      expect(dbPostProcess(high, identityGeo).quads, hasLength(1));
    });
  });

  group('unclip ratio 1.5', () {
    test('a known rect grows by d = area*1.5/perimeter per side', () {
      final map = emptyMap();
      // 100x40 region (boundary pixels span 99x39 — the fitted rect measures
      // pixel CENTERS, so expected fitted w=99, h=39).
      fillRect(map, const Rect.fromLTWH(300, 300, 100, 40), 0.9);
      final result = dbPostProcess(map, identityGeo);
      expect(result.quads, hasLength(1));

      const w = 99.0, h = 39.0;
      const d = (w * h) * kDbUnclipRatio / (2 * (w + h));
      final bbox = result.quads.first.boundingBox;
      expect(bbox.width, closeTo(w + 2 * d, 1.5));
      expect(bbox.height, closeTo(h + 2 * d, 1.5));
      // And it grew, meaningfully (d ≈ 21 px).
      expect(bbox.width, greaterThan(100));
    });
  });

  group('connected components (8-connectivity)', () {
    test('two separate blobs produce two quads', () {
      final map = emptyMap();
      fillRect(map, const Rect.fromLTWH(50, 50, 60, 20), 0.9);
      fillRect(map, const Rect.fromLTWH(300, 300, 60, 20), 0.9);
      expect(dbPostProcess(map, identityGeo).quads, hasLength(2));
    });

    test('diagonally-adjacent pixels join one component (8-conn)', () {
      final map = emptyMap();
      // Two 10x10 blocks touching only at one diagonal corner.
      fillRect(map, const Rect.fromLTWH(100, 100, 10, 10), 0.9);
      fillRect(map, const Rect.fromLTWH(110, 110, 10, 10), 0.9);
      expect(dbPostProcess(map, identityGeo).quads, hasLength(1));
    });

    test('sub-minimum boxes (short side < 3px) are dropped as noise', () {
      final map = emptyMap();
      fillRect(map, const Rect.fromLTWH(100, 100, 40, 2), 0.9);
      expect(dbPostProcess(map, identityGeo).quads, isEmpty);
    });
  });

  group('min-area ROTATED rect', () {
    test('a 45-degree bar yields a tilted quad, not its huge AABB', () {
      final map = emptyMap();
      // Diagonal bar: thick 45° line from (200,200) to (320,320).
      for (var t = 0; t < 120; t++) {
        for (var k = -6; k <= 6; k++) {
          final x = 200 + t + k;
          final y = 200 + t - k;
          if (x >= 0 && x < size && y >= 0 && y < size) {
            map[y * size + x] = 0.95;
          }
        }
      }
      final result = dbPostProcess(map, identityGeo);
      expect(result.quads, hasLength(1));
      final quad = result.quads.first;

      // The min-area rect of a 45° bar is much smaller than its bounding
      // box; a non-rotated fit would return the AABB itself.
      final aabb = quad.boundingBox;
      expect(quad.area, lessThan(aabb.width * aabb.height * 0.75));

      // Its long edge runs at ~45°.
      final e = quad.tr - quad.tl;
      final angle = math.atan2(e.dy.abs(), e.dx.abs()) * 180 / math.pi;
      expect(angle, closeTo(45, 6));
    });
  });

  group('coordinate space: quads come back in FRAME space', () {
    test('with a real letterbox geometry the quad center un-maps', () {
      // 1280x720 frame -> scale 0.575, padY 161.
      final geo = computeLetterboxGeometry(1280, 720);
      final map = emptyMap();
      // Draw the region where a frame-space rect at (400..560, 300..360)
      // lands in model space: x*0.575, y*0.575+161.
      final modelRect = Rect.fromLTRB(
        400 * geo.scale + geo.padX,
        300 * geo.scale + geo.padY,
        560 * geo.scale + geo.padX,
        360 * geo.scale + geo.padY,
      );
      fillRect(map, modelRect, 0.9);
      final result = dbPostProcess(map, geo);
      expect(result.quads, hasLength(1));
      final c = result.quads.first.center;
      // Center back in FRAME coordinates (~480, ~330), well outside model
      // pixel range for y (which would be ~350 in 736-space).
      expect(c.dx, closeTo(480, 12));
      expect(c.dy, closeTo(330, 14));
    });
  });

  group('reading order (top->bottom bands, then left->right)', () {
    test('three regions sort into reading order', () {
      final map = emptyMap();
      fillRect(map, const Rect.fromLTWH(400, 100, 60, 20), 0.9); // top-right
      fillRect(map, const Rect.fromLTWH(100, 104, 60, 20), 0.9); // top-left
      fillRect(map, const Rect.fromLTWH(200, 400, 60, 20), 0.9); // bottom
      final quads = dbPostProcess(map, identityGeo).quads;
      expect(quads, hasLength(3));
      // Band 1: left then right; band 2: the bottom one.
      expect(quads[0].center.dx, lessThan(quads[1].center.dx));
      expect(quads[0].center.dy, closeTo(quads[1].center.dy, 30));
      expect(quads[2].center.dy, greaterThan(300));
    });

    test('sortReadingOrder is stable for an empty/single list', () {
      expect(sortReadingOrder([]), isEmpty);
      final q = [Quad.fromRect(const Rect.fromLTWH(0, 0, 10, 10))];
      expect(sortReadingOrder(q), hasLength(1));
    });
  });
}
