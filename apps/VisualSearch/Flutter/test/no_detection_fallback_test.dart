import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:snapseek/services/postprocessor.dart';
import 'package:snapseek/services/preprocessor.dart';

import 'test_helpers.dart';

void main() {
  group('no-detection fallback (embed is NEVER skipped)', () {
    test('all-zero output yields no boxes and no primary', () {
      final dets = postprocessDetect(identityRequest(emptyOutput()));
      expect(dets, isEmpty);
      expect(primaryBox(dets), isNull);
    });

    test('sub-threshold noise everywhere yields no primary', () {
      final out = emptyOutput();
      const n = kNumAnchors;
      for (var i = 0; i < n; i++) {
        out[(4 + kClassPerson) * n + i] = 0.10; // all below 0.25
      }
      final dets = postprocessDetect(identityRequest(out));
      expect(primaryBox(dets), isNull);
    });

    test('null primary → preprocessEmbed still produces a valid embed tensor '
        'via center-crop', () {
      final rgb = Uint8List(300 * 300 * 3)..fillRange(0, 300 * 300 * 3, 90);
      final bundle = preprocessEmbed(EmbedRequest(
        width: 300,
        height: 300,
        rgb: rgb,
        box: null, // the fallback path
      ));
      expect(bundle.input.length, 3 * kEmbedSize * kEmbedSize);
      // Centered square crop of a 300x300 frame is the whole frame.
      expect(bundle.cropRect.width, closeTo(300, 1e-6));
      expect(bundle.cropRect.height, closeTo(300, 1e-6));
    });
  });
}
