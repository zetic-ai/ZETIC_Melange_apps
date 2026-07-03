import 'package:flutter_test/flutter_test.dart';
import 'package:siteguard/models/detection.dart';
import 'package:siteguard/services/postprocessor.dart';

import 'test_helpers.dart';

void main() {
  group('channel-major [1,17,8400] decode', () {
    test('one hand-built anchor decodes to exactly one correct detection', () {
      final out = emptyOutput();
      // Anchor 100: a 100x50 hardhat centered at (320,160) in 640-space.
      setAnchor(out, 100,
          cx: 320, cy: 160, w: 100, h: 50, scores: {kClassHardhat: 0.9});

      final dets = postprocessOutput(identityRequest(out));

      expect(dets, hasLength(1));
      final d = dets.single;
      expect(d.classId, kClassHardhat);
      expect(d.confidence, closeTo(0.9, 1e-6));
      // (320-50)/640 .. (320+50)/640 x, (160-25)/640 .. (160+25)/640 y.
      expect(d.rect.left, closeTo(270 / 640, 1e-6));
      expect(d.rect.top, closeTo(135 / 640, 1e-6));
      expect(d.rect.right, closeTo(370 / 640, 1e-6));
      expect(d.rect.bottom, closeTo(185 / 640, 1e-6));
    });

    test('high-index anchor is read with anchor stride, not channel stride',
        () {
      final out = emptyOutput();
      // Anchor 8399 (the last one) — a misread of the layout (row-major
      // [8400,17] instead of channel-major [17,8400]) would scatter these
      // floats across unrelated anchors/channels and produce garbage or
      // nothing at this exact location.
      setAnchor(out, kNumAnchors - 1,
          cx: 100, cy: 500, w: 40, h: 80, scores: {kClassVest: 0.7});

      final dets = postprocessOutput(identityRequest(out));

      expect(dets, hasLength(1));
      final d = dets.single;
      expect(d.classId, kClassVest);
      expect(d.confidence, closeTo(0.7, 1e-6));
      expect(d.rect.left, closeTo(80 / 640, 1e-6));
      expect(d.rect.top, closeTo(460 / 640, 1e-6));
      expect(d.rect.right, closeTo(120 / 640, 1e-6));
      expect(d.rect.bottom, closeTo(540 / 640, 1e-6));
    });

    test('two anchors on different classes both decode', () {
      final out = emptyOutput();
      setAnchor(out, 10,
          cx: 100, cy: 100, w: 60, h: 60, scores: {kClassHardhat: 0.8});
      setAnchor(out, 5000,
          cx: 500, cy: 400, w: 90, h: 120, scores: {kClassNoVest: 0.55});

      final dets = postprocessOutput(identityRequest(out));

      expect(dets, hasLength(2));
      expect(dets.map((d) => d.classId).toSet(),
          {kClassHardhat, kClassNoVest});
    });

    test('wrong-length buffer trips the assert in debug mode', () {
      expect(
        () => postprocessOutput(
          PostprocessRequest(
            output: emptyOutput().sublist(0, 17),
            scale: 1,
            padX: 0,
            padY: 0,
            srcWidth: 640,
            srcHeight: 640,
          ),
        ),
        throwsA(isA<AssertionError>()),
      );
    });
  });
}
