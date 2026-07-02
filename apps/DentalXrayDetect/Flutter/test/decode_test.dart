import 'dart:typed_data';

import 'package:dentalxraydetect/models/detection.dart';
import 'package:dentalxraydetect/services/postprocessor.dart';
import 'package:dentalxraydetect/services/preprocessor.dart';
import 'package:flutter_test/flutter_test.dart';

/// Sets a value at (channel, anchor) -> index `channel * numAnchors + anchor`.
typedef OutputSetter = void Function(int channel, int anchor, double value);

/// Build a zeroed channel-major [1,7,8400] output, with `set` letting tests
/// place a value at a given (channel, anchor).
Float32List buildOutput(void Function(OutputSetter set) build) {
  final Float32List out = Float32List(kNumChannels * kNumAnchors);
  build((int channel, int anchor, double value) {
    out[channel * kNumAnchors + anchor] = value;
  });
  return out;
}

/// Identity letterbox: source already 640x640, so model space == source space.
LetterboxParams identityLetterbox() => computeLetterbox(640, 640);

void main() {
  group('decode [1,7,8400]', () {
    test('test_channel_major_decode reads stride-across-anchors, not the 7', () {
      const int anchor = 100;
      const int cls = 2; // impacted_tooth
      final Float32List out = buildOutput((OutputSetter set) {
        // Box channels 0..3 at the anchor (640 pixel space).
        set(0, anchor, 320); // cx
        set(1, anchor, 320); // cy
        set(2, anchor, 100); // w
        set(3, anchor, 80); // h
        // Class score on channel 4+cls.
        set(4 + cls, anchor, 0.9);
      });

      final List<Detection> dets = decodeDetections(out, identityLetterbox());

      expect(dets.length, 1);
      final Detection d = dets.first;
      expect(d.classId, cls);
      expect(d.confidence, closeTo(0.9, 1e-6));
      // cxcywh(320,320,100,80) -> xyxy(270,280,370,360).
      expect(d.left, closeTo(270, 1e-3));
      expect(d.top, closeTo(280, 1e-3));
      expect(d.right, closeTo(370, 1e-3));
      expect(d.bottom, closeTo(360, 1e-3));
    });

    test('a row-major (transposed) read fails: 7-contiguous layout yields no box',
        () {
      // Same logical values but written 7-contiguous per anchor (the classic
      // bug). The channel-major decoder must NOT interpret this as a detection
      // at the intended anchor: the class score lands on a box channel of some
      // other anchor and the intended anchor's class channels stay 0.
      const int anchor = 100;
      final Float32List rowMajor = Float32List(kNumChannels * kNumAnchors);
      final List<double> vals = <double>[320, 320, 100, 80, 0, 0, 0.9];
      for (int c = 0; c < kNumChannels; c++) {
        rowMajor[anchor * kNumChannels + c] = vals[c];
      }
      final List<Detection> dets =
          decodeDetections(rowMajor, identityLetterbox());
      // The single 0.9 written row-major does not sit on a class channel of a
      // decoded anchor with a plausible box, so no valid detection at anchor 100.
      expect(dets.where((Detection d) => d.classId == 2 && d.confidence > 0.85),
          isEmpty);
    });

    test('test_max_over_3_class picks argmax over channels 4..6', () {
      const int anchor = 50;
      final Float32List out = buildOutput((OutputSetter set) {
        set(0, anchor, 300);
        set(1, anchor, 300);
        set(2, anchor, 60);
        set(3, anchor, 60);
        set(4 + 0, anchor, 0.40); // caries
        set(4 + 2, anchor, 0.60); // impacted_tooth (higher)
      });

      final List<Detection> dets = decodeDetections(out, identityLetterbox());

      expect(dets.length, 1);
      expect(dets.first.classId, 2, reason: 'argmax class is impacted_tooth(2)');
      expect(dets.first.confidence, closeTo(0.60, 1e-6));
    });

    test('test_no_double_sigmoid treats class channels as final probabilities',
        () {
      // Class scores are ALREADY sigmoid-activated in the ONNX graph. A true
      // 0.80 must be consumed AS-IS (not sigmoid(0.80)=0.690), and a raw 0.42
      // (< 0.45) must be DROPPED — whereas a wrongly re-applied sigmoid would
      // lift sigmoid(0.42)=0.603 over the gate and add a phantom detection.
      final Float32List out = buildOutput((OutputSetter set) {
        // Anchor A: passes, high confidence.
        set(0, 0, 200);
        set(1, 0, 200);
        set(2, 0, 40);
        set(3, 0, 40);
        set(4 + 0, 0, 0.80); // caries @ 0.80
        // Anchor B: raw 0.42, below the 0.45 gate.
        set(0, 1, 500);
        set(1, 1, 500);
        set(2, 1, 40);
        set(3, 1, 40);
        set(4 + 2, 1, 0.42); // impacted_tooth @ 0.42
      });

      final List<Detection> dets = decodeDetections(out, identityLetterbox());

      expect(dets.length, 1, reason: 'raw 0.42 must be dropped (no re-sigmoid)');
      expect(dets.first.classId, 0);
      expect(dets.first.confidence, closeTo(0.80, 1e-6),
          reason: 'confidence must be raw 0.80, not sigmoid(0.80)=0.690');
    });

    test('test_coordinate_space_pixel_not_normalized keeps 640-pixel centers',
        () {
      const int anchor = 7;
      final Float32List out = buildOutput((OutputSetter set) {
        set(0, anchor, 320); // cx in PIXELS (==640/2), not 0.5 normalized
        set(1, anchor, 160);
        set(2, anchor, 100);
        set(3, anchor, 100);
        set(4 + 1, anchor, 0.7);
      });

      final Detection d = decodeDetections(out, identityLetterbox()).first;
      // Center must remain at pixel 320/160, proving no normalization.
      expect((d.left + d.right) / 2, closeTo(320, 1e-3));
      expect((d.top + d.bottom) / 2, closeTo(160, 1e-3));
    });
  });
}
