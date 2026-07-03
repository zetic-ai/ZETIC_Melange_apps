import 'dart:typed_data';
import 'dart:ui' show Offset, Size;

import 'package:flutter_test/flutter_test.dart';
import 'package:signtranslate/models/text_region.dart';
import 'package:signtranslate/services/coordinate_mapper.dart';
import 'package:signtranslate/services/detector_preprocessor.dart';
import 'package:signtranslate/services/frame_data.dart';

void main() {
  group('buffer-orientation transform round-trips (PyroGuard trap: the bug '
      'can be a SPURIOUS rotation, not a missing one)', () {
    for (final rot in [0, 90, 180, 270]) {
      test('rotation $rot: uprightToRaw ∘ rawToUpright = identity', () {
        const rawW = 64, rawH = 48;
        for (final (rx, ry) in [(0, 0), (63, 47), (10, 33), (40, 7)]) {
          final (ux, uy) = rawToUpright(rx, ry, rawW, rawH, rot);
          final (backX, backY) = uprightToRaw(ux, uy, rawW, rawH, rot);
          expect((backX, backY), (rx, ry),
              reason: 'raw ($rx,$ry) failed round-trip at rot=$rot');
          // Upright coords stay inside the upright frame.
          final swap = rot == 90 || rot == 270;
          expect(ux, inInclusiveRange(0, (swap ? rawH : rawW) - 1));
          expect(uy, inInclusiveRange(0, (swap ? rawW : rawH) - 1));
        }
      });
    }

    test('rotation 0 is a true no-op (iOS upright-buffer case)', () {
      expect(uprightToRaw(12, 34, 100, 80, 0), (12, 34));
      expect(rawToUpright(12, 34, 100, 80, 0), (12, 34));
    });

    test('rotation 90 (Android sensor case): a marked raw pixel lands at '
        'the expected upright position end-to-end through BGR conversion',
        () {
      // Raw landscape 8x6 BGRA buffer; mark raw pixel (7, 0) pure blue.
      const rawW = 8, rawH = 6;
      final bgra = Uint8List(rawW * rawH * 4);
      final idx = (0 * rawW + 7) * 4;
      bgra[idx] = 255; // B

      final frame = FrameData.bgra8888(
        width: rawW,
        height: rawH,
        bgra: bgra,
        bgraRowStride: rawW * 4,
        rotationDegrees: 90,
      );
      final upright = convertToUprightBgr(frame);
      // Upright frame is 6 wide, 8 tall.
      expect(upright.width, rawH);
      expect(upright.height, rawW);

      // rawToUpright(7,0, 8,6, 90) = (rawW-1-ry, rx) = (7-0? no:
      // (rawW - 1 - ry, rx) with rawW=8 -> (8-1-0, 7) = (7,7)?  — compute
      // via the function itself to stay definition-consistent:
      final (ux, uy) = rawToUpright(7, 0, rawW, rawH, 90);
      final blue = upright.bgr[(uy * upright.width + ux) * 3];
      expect(blue, 255,
          reason: 'marked pixel must land at the mapped upright position');

      // And nowhere else: total blue mass equals exactly one pixel.
      var total = 0;
      for (var i = 0; i < upright.width * upright.height; i++) {
        total += upright.bgr[i * 3];
      }
      expect(total, 255);
    });
  });

  group('overlay mapping (BoxFit.cover), pure function frame -> screen', () {
    test('upright portrait buffer 720x1280 on a 390x844 screen', () {
      const screen = Size(390, 844);
      // cover scale = max(390/720, 844/1280) = 0.659375; dx = (390-720*s)/2.
      const s = 844 / 1280;
      final center = mapFrameToScreen(
          const Offset(360, 640), 720, 1280, screen);
      expect(center.dx, closeTo(390 / 2, 1e-6));
      expect(center.dy, closeTo(844 / 2, 1e-6));

      final corner = mapFrameToScreen(Offset.zero, 720, 1280, screen);
      expect(corner.dy, closeTo(0, 1e-6));
      expect(corner.dx, closeTo((390 - 720 * s) / 2, 1e-6));
      expect(corner.dx, lessThan(0)); // cover crops horizontally
    });

    test('landscape buffer 1280x720 on a portrait screen crops vertically',
        () {
      const screen = Size(390, 844);
      // cover scale = max(390/1280, 844/720) = 844/720.
      final corner = mapFrameToScreen(Offset.zero, 1280, 720, screen);
      expect(corner.dx, lessThan(0));
      expect(corner.dy, closeTo(0, 1e-6));
      final p = mapFrameToScreen(const Offset(640, 360), 1280, 720, screen);
      expect(p.dx, closeTo(390 / 2, 1e-6));
      expect(p.dy, closeTo(844 / 2, 1e-6));
    });

    test('mapFrameToScreen ∘ mapScreenToFrame = identity (round-trip)', () {
      const screen = Size(393, 852);
      for (final p in [
        const Offset(0, 0),
        const Offset(719, 1279),
        const Offset(123.4, 567.8),
      ]) {
        final back = mapScreenToFrame(
          mapFrameToScreen(p, 720, 1280, screen),
          720,
          1280,
          screen,
        );
        expect((back - p).distance, lessThan(1e-6));
      }
    });

    test('a known frame quad maps to screen preserving relative geometry',
        () {
      const screen = Size(400, 800);
      final quad = Quad(
        const Offset(100, 200),
        const Offset(300, 200),
        const Offset(300, 260),
        const Offset(100, 260),
      );
      final mapped = quad.map(
        (p) => mapFrameToScreen(p, 720, 1280, screen),
      );
      // Aspect preserved: width/height ratio unchanged under uniform scale.
      final wRatio = (mapped.tr.dx - mapped.tl.dx) /
          (mapped.bl.dy - mapped.tl.dy);
      expect(wRatio, closeTo(200 / 60, 1e-6));
      // NO rotation: top edge stays horizontal (the PyroGuard bug would
      // transpose it).
      expect(mapped.tl.dy, closeTo(mapped.tr.dy, 1e-9));
    });
  });
}
