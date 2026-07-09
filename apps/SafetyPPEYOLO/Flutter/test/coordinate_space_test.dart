import 'package:flutter_test/flutter_test.dart';
import 'package:siteguard/models/detection.dart';
import 'package:siteguard/services/postprocessor.dart';

import 'test_helpers.dart';

void main() {
  group('coordinate space semantics', () {
    test('decode treats box coords as 640-space PIXELS, not normalized 0..1',
        () {
      final out = emptyOutput();
      // cx=320 (pixels) must land at the horizontal center of the frame.
      // If the decoder wrongly treated coords as normalized (320 -> clamped
      // 1.0), the rect would collapse to the right edge instead.
      setAnchor(out, 7,
          cx: 320, cy: 320, w: 640, h: 640, scores: {kClassHardhat: 0.9});

      final dets = postprocessOutput(identityRequest(out));

      expect(dets, hasLength(1));
      final r = dets.single.rect;
      expect(r.center.dx, closeTo(0.5, 1e-6));
      expect(r.center.dy, closeTo(0.5, 1e-6));
      expect(r.width, closeTo(1.0, 1e-6));
      expect(r.height, closeTo(1.0, 1e-6));
    });

    test('emitted rects are normalized 0..1 and clamped to frame bounds', () {
      final out = emptyOutput();
      // Box partially outside the frame must clamp, not go negative.
      setAnchor(out, 8,
          cx: 10, cy: 10, w: 100, h: 100, scores: {kClassVest: 0.5});

      final dets = postprocessOutput(identityRequest(out));

      expect(dets, hasLength(1));
      final r = dets.single.rect;
      expect(r.left, 0.0);
      expect(r.top, 0.0);
      expect(r.right, closeTo(60 / 640, 1e-6));
      expect(r.bottom, closeTo(60 / 640, 1e-6));
    });

    test('degenerate (fully out-of-frame) box is dropped, not emitted', () {
      final out = emptyOutput();
      setAnchor(out, 9,
          cx: -100, cy: -100, w: 50, h: 50, scores: {kClassVest: 0.5});

      final dets = postprocessOutput(identityRequest(out));
      expect(dets, isEmpty);
    });
  });
}
