import 'dart:typed_data';

import 'package:dentalxraydetect/services/postprocessor.dart';
import 'package:dentalxraydetect/services/preprocessor.dart';
import 'package:flutter_test/flutter_test.dart';

/// Confidence-threshold boundary at 0.45.
///
/// The decoder uses STRICT `>` to match the Python validation harness
/// (`m = conf > conf_thres`) so the Dart pipeline reproduces the measured demo
/// detections exactly. Therefore a score of EXACTLY 0.45 is DROPPED. (SPEC prose
/// says "≥ 0.45"; the harness is the source of truth for reproduction and uses
/// strict `>`, so this test pins strict `>`.)
void main() {
  LetterboxParams identity() => computeLetterbox(640, 640);

  Float32List oneAnchor(double score) {
    final Float32List out = Float32List(kNumChannels * kNumAnchors);
    void set(int c, int a, double v) => out[c * kNumAnchors + a] = v;
    set(0, 0, 200);
    set(1, 0, 200);
    set(2, 0, 20);
    set(3, 0, 20);
    set(4 + 0, 0, score); // caries
    return out;
  }

  group('threshold boundary @ 0.45 (strict >)', () {
    test('just-above 0.45 is kept', () {
      expect(decodeDetections(oneAnchor(0.46), identity()).length, 1);
    });

    test('just-below 0.45 is dropped', () {
      expect(decodeDetections(oneAnchor(0.44), identity()).length, 0);
    });

    test('exactly 0.45 is dropped (strict >, harness-exact)', () {
      expect(decodeDetections(oneAnchor(0.45), identity()).length, 0);
    });
  });
}
