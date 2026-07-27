import 'package:flutter_test/flutter_test.dart';
import 'package:snapseek/services/postprocessor.dart';
import 'package:snapseek/services/preprocessor.dart';

import 'test_helpers.dart';

/// SnapSeek captures via takePicture() and bakes EXIF orientation, so the frame
/// reaching the pipeline is ALREADY upright — the chosen transform applies NO
/// rotation (deliberate deviation from PyroGuard's live stream; see
/// preprocessor.dart). This test asserts that for the believed orientation
/// (upright portrait) a known box round-trips WITHOUT being transposed — the
/// classic orientation bug turns a wide box into a tall sliver.
void main() {
  group('orientation — upright portrait, no spurious rotation', () {
    test('a WIDE box in a portrait frame stays wide after round-trip', () {
      const srcW = 1080, srcH = 1920; // portrait, upright
      final g = letterboxGeometry(srcW, srcH);

      // A deliberately WIDE box (w > h) in source pixels.
      const sx1 = 100.0, sy1 = 800.0, sx2 = 900.0, sy2 = 1100.0;

      final out = emptyOutput();
      final lx1 = sx1 * g.scale + g.padX;
      final ly1 = sy1 * g.scale + g.padY;
      final lx2 = sx2 * g.scale + g.padX;
      final ly2 = sy2 * g.scale + g.padY;
      setAnchor(out, 3,
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
      final pxW = r.width * srcW;
      final pxH = r.height * srcH;
      // Round-trips to the same box AND stays wider than tall (not transposed).
      expect(pxW, closeTo(800, 1e-2));
      expect(pxH, closeTo(300, 1e-2));
      expect(pxW, greaterThan(pxH));
    });
  });
}
