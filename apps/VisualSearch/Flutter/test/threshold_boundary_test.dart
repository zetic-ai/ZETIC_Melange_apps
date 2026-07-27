import 'package:flutter_test/flutter_test.dart';
import 'package:snapseek/services/postprocessor.dart';

import 'test_helpers.dart';

void main() {
  group('confidence threshold boundary (default 0.25)', () {
    test('just-above threshold is kept', () {
      final out = emptyOutput();
      setAnchor(out, 10,
          cx: 300, cy: 300, w: 50, h: 50, scores: {kClassPerson: 0.26});
      final dets = postprocessDetect(identityRequest(out));
      expect(dets, hasLength(1));
    });

    test('just-below threshold is dropped', () {
      final out = emptyOutput();
      setAnchor(out, 10,
          cx: 300, cy: 300, w: 50, h: 50, scores: {kClassPerson: 0.24});
      final dets = postprocessDetect(identityRequest(out));
      expect(dets, isEmpty);
    });

    test('exactly at threshold is dropped (strict >)', () {
      final out = emptyOutput();
      setAnchor(out, 10,
          cx: 300, cy: 300, w: 50, h: 50, scores: {kClassPerson: 0.25});
      final dets = postprocessDetect(identityRequest(out));
      expect(dets, isEmpty);
    });

    test('a custom higher threshold filters a mid-confidence box', () {
      final out = emptyOutput();
      setAnchor(out, 10,
          cx: 300, cy: 300, w: 50, h: 50, scores: {kClassPerson: 0.40});
      expect(postprocessDetect(identityRequest(out, conf: 0.35)), hasLength(1));
      expect(postprocessDetect(identityRequest(out, conf: 0.45)), isEmpty);
    });
  });
}
