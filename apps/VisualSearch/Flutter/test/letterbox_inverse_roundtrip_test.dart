import 'package:flutter_test/flutter_test.dart';
import 'package:snapseek/services/postprocessor.dart';
import 'package:snapseek/services/preprocessor.dart';

import 'test_helpers.dart';

void main() {
  group('letterbox inverse round-trip', () {
    for (final (srcW, srcH) in [
      (1280, 720),
      (720, 1280),
      (1920, 1080),
      (800, 800),
    ]) {
      test('known box round-trips on ${srcW}x$srcH within tolerance', () {
        // Forward geometry from the SAME code the app uses.
        final g = letterboxGeometry(srcW, srcH);

        // A known box in SOURCE pixels.
        const sx1 = 200.0, sy1 = 150.0, sx2 = 500.0, sy2 = 400.0;

        // Forward: source → 640 letterbox (scale THEN pad).
        final lx1 = sx1 * g.scale + g.padX;
        final ly1 = sy1 * g.scale + g.padY;
        final lx2 = sx2 * g.scale + g.padX;
        final ly2 = sy2 * g.scale + g.padY;

        final out = emptyOutput();
        setAnchor(out, 7,
            cx: (lx1 + lx2) / 2,
            cy: (ly1 + ly2) / 2,
            w: lx2 - lx1,
            h: ly2 - ly1,
            scores: {kClassPerson: 0.9});

        final dets = postprocessDetect(PostprocessRequest(
          output: out,
          scale: g.scale,
          padX: g.padX,
          padY: g.padY,
          srcWidth: srcW,
          srcHeight: srcH,
        ));

        expect(dets, hasLength(1));
        final r = dets.single.rect;
        expect(r.left * srcW, closeTo(sx1, 1e-3));
        expect(r.top * srcH, closeTo(sy1, 1e-3));
        expect(r.right * srcW, closeTo(sx2, 1e-3));
        expect(r.bottom * srcH, closeTo(sy2, 1e-3));
      });
    }

    test('geometry constants: portrait 720x1280 pads x, not y', () {
      final g = letterboxGeometry(720, 1280);
      expect(g.scale, closeTo(0.5, 1e-9)); // 640/1280
      expect(g.padX, (640 - 360) ~/ 2); // 140
      expect(g.padY, 0);
    });
  });
}
