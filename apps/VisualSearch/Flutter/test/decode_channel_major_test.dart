import 'package:flutter_test/flutter_test.dart';
import 'package:snapseek/services/postprocessor.dart';

import 'test_helpers.dart';

void main() {
  group('channel-major [1,84,8400] decode', () {
    test('one hand-built anchor decodes to exactly one correct detection', () {
      final out = emptyOutput();
      // Anchor 100: a 100x50 box centered at (320,160) in 640-space, person.
      setAnchor(out, 100,
          cx: 320, cy: 160, w: 100, h: 50, scores: {kClassPerson: 0.9});

      final dets = postprocessDetect(identityRequest(out));

      expect(dets, hasLength(1));
      final d = dets.single;
      expect(d.classId, kClassPerson);
      expect(d.confidence, closeTo(0.9, 1e-6));
      expect(d.rect.left, closeTo(270 / 640, 1e-6));
      expect(d.rect.top, closeTo(135 / 640, 1e-6));
      expect(d.rect.right, closeTo(370 / 640, 1e-6));
      expect(d.rect.bottom, closeTo(185 / 640, 1e-6));
    });

    test('high-index anchor is read with anchor stride, not channel stride', () {
      final out = emptyOutput();
      // A row-major misread ([8400,84]) would scatter these floats and produce
      // garbage at this location.
      setAnchor(out, kNumAnchors - 1,
          cx: 100, cy: 500, w: 40, h: 80, scores: {kClassBottle: 0.7});

      final dets = postprocessDetect(identityRequest(out));

      expect(dets, hasLength(1));
      final d = dets.single;
      expect(d.classId, kClassBottle);
      expect(d.confidence, closeTo(0.7, 1e-6));
      expect(d.rect.left, closeTo(80 / 640, 1e-6));
      expect(d.rect.top, closeTo(460 / 640, 1e-6));
      expect(d.rect.right, closeTo(120 / 640, 1e-6));
      expect(d.rect.bottom, closeTo(540 / 640, 1e-6));
    });

    test('per-anchor max over 80 classes picks the argmax class', () {
      final out = emptyOutput();
      setAnchor(out, 42, cx: 200, cy: 200, w: 60, h: 60, scores: {
        kClassPerson: 0.30,
        kClassBackpack: 0.81, // the winner
        kClassBottle: 0.55,
      });

      final dets = postprocessDetect(identityRequest(out));
      expect(dets, hasLength(1));
      expect(dets.single.classId, kClassBackpack);
      expect(dets.single.confidence, closeTo(0.81, 1e-6));
    });

    test('wrong-length buffer trips the assert in debug mode', () {
      expect(
        () => postprocessDetect(PostprocessRequest(
          output: emptyOutput().sublist(0, 84),
          scale: 1,
          padX: 0,
          padY: 0,
          srcWidth: 640,
          srcHeight: 640,
        )),
        throwsA(isA<AssertionError>()),
      );
    });
  });
}
