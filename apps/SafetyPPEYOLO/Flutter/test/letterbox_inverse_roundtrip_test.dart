import 'package:flutter_test/flutter_test.dart';
import 'package:siteguard/models/detection.dart';
import 'package:siteguard/services/postprocessor.dart';
import 'package:siteguard/services/preprocessor.dart';

import 'test_helpers.dart';

/// Forward letterbox for a source box, mirroring the preprocessor's geometry
/// (scale = min(640/w, 640/h), round, //2-centered padding).
({double scale, int padX, int padY}) letterboxGeom(int srcW, int srcH) {
  const size = kInputSize;
  final double scale =
      (size / srcW) < (size / srcH) ? (size / srcW) : (size / srcH);
  final int newW = (srcW * scale).round();
  final int newH = (srcH * scale).round();
  return (
    scale: scale,
    padX: (size - newW) ~/ 2,
    padY: (size - newH) ~/ 2,
  );
}

void main() {
  group('letterbox inverse round-trip', () {
    for (final (srcW, srcH) in [(1280, 720), (720, 1280), (1920, 1080)]) {
      test('known box round-trips on ${srcW}x$srcH within tolerance', () {
        final g = letterboxGeom(srcW, srcH);

        // A known box in SOURCE pixels.
        const sx1 = 200.0, sy1 = 150.0, sx2 = 500.0, sy2 = 400.0;

        // Forward: source -> 640 letterbox space (exact preprocessor order:
        // scale THEN pad).
        final lx1 = sx1 * g.scale + g.padX;
        final ly1 = sy1 * g.scale + g.padY;
        final lx2 = sx2 * g.scale + g.padX;
        final ly2 = sy2 * g.scale + g.padY;

        // Feed through the real postprocessor as a cxcywh anchor.
        final out = emptyOutput();
        setAnchor(out, 42,
            cx: (lx1 + lx2) / 2,
            cy: (ly1 + ly2) / 2,
            w: lx2 - lx1,
            h: ly2 - ly1,
            scores: {kClassHardhat: 0.9});

        final dets = postprocessOutput(PostprocessRequest(
          output: out,
          scale: g.scale,
          padX: g.padX,
          padY: g.padY,
          srcWidth: srcW,
          srcHeight: srcH,
        ));

        expect(dets, hasLength(1));
        final r = dets.single.rect;
        // Inverse must return the original source box (normalized), within
        // float tolerance.
        expect(r.left * srcW, closeTo(sx1, 1e-3));
        expect(r.top * srcH, closeTo(sy1, 1e-3));
        expect(r.right * srcW, closeTo(sx2, 1e-3));
        expect(r.bottom * srcH, closeTo(sy2, 1e-3));
      });
    }

    test('geometry constants: portrait 720x1280 pads x, not y', () {
      final g = letterboxGeom(720, 1280);
      expect(g.scale, closeTo(0.5, 1e-9)); // 640/1280
      expect(g.padX, (640 - 360) ~/ 2); // 140
      expect(g.padY, 0);
    });
  });
}
