import 'dart:typed_data';

import 'package:aerialdetect/models/detection.dart';
import 'package:aerialdetect/services/postprocessor.dart';
import 'package:aerialdetect/services/preprocessor.dart';
import 'package:flutter_test/flutter_test.dart';

/// Sets a value at (channel, anchor) -> index `channel * numAnchors + anchor`.
typedef OutputSetter = void Function(int channel, int anchor, double value);

/// Build a zeroed channel-major [1,14,17661] output, with `set` letting tests
/// place a value at a given (channel, anchor).
Float32List buildOutput(void Function(OutputSetter set) build) {
  final Float32List out = Float32List(kNumChannels * kNumAnchors);
  build((int channel, int anchor, double value) {
    out[channel * kNumAnchors + anchor] = value;
  });
  return out;
}

/// Identity letterbox: source already 928x928, so model space == source space.
LetterboxParams identityLetterbox() => computeLetterbox(928, 928);

void main() {
  group('decode [1,14,17661]', () {
    test('test_channel_major_decode reads stride-across-anchors, not the 14',
        () {
      const int anchor = 100;
      const int cls = 3; // car
      final Float32List out = buildOutput((set) {
        // Box channels 0..3 at the anchor (928 pixel space).
        set(0, anchor, 464); // cx
        set(1, anchor, 464); // cy
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
      // cxcywh(464,464,100,80) -> xyxy(414,424,514,504).
      expect(d.left, closeTo(414, 1e-3));
      expect(d.top, closeTo(424, 1e-3));
      expect(d.right, closeTo(514, 1e-3));
      expect(d.bottom, closeTo(504, 1e-3));
    });

    test('test_no_double_sigmoid treats class channels as final probabilities',
        () {
      // Channel scores are ALREADY sigmoid-activated in the ONNX graph
      // (/model.22/Sigmoid). Anchor A=0.30 must pass and stay 0.30; anchor
      // B=0.20 must be dropped. A wrongly re-applied sigmoid would make
      // sigmoid(0.20)=0.55 pass (2 dets) and report 0.30 as 0.574.
      final Float32List out = buildOutput((set) {
        // Anchor A: passes.
        set(0, 0, 200);
        set(1, 0, 200);
        set(2, 0, 40);
        set(3, 0, 40);
        set(4 + 3, 0, 0.30); // car @ 0.30
        // Anchor B: below threshold.
        set(0, 1, 600);
        set(1, 1, 600);
        set(2, 1, 40);
        set(3, 1, 40);
        set(4 + 5, 1, 0.20); // truck @ 0.20
      });

      final List<Detection> dets = decodeDetections(out, identityLetterbox());

      expect(dets.length, 1, reason: '0.20 must be dropped (no re-sigmoid)');
      expect(dets.first.classId, 3);
      expect(dets.first.confidence, closeTo(0.30, 1e-6),
          reason: 'confidence must be raw 0.30, not sigmoid(0.30)=0.574');
    });

    test('test_max_over_class_channels picks argmax over channels 4..13', () {
      const int anchor = 50;
      final Float32List out = buildOutput((set) {
        set(0, anchor, 300);
        set(1, anchor, 300);
        set(2, anchor, 60);
        set(3, anchor, 60);
        set(4 + 2, anchor, 0.40); // bicycle
        set(4 + 8, anchor, 0.60); // bus (higher)
      });

      final List<Detection> dets = decodeDetections(out, identityLetterbox());

      expect(dets.length, 1);
      expect(dets.first.classId, 8, reason: 'argmax class is bus(8)');
      expect(dets.first.confidence, closeTo(0.60, 1e-6));
    });

    test('test_threshold_boundary keeps just-above, drops at/just-below (>)',
        () {
      Float32List one(double score) => buildOutput((set) {
            set(0, 0, 200);
            set(1, 0, 200);
            set(2, 0, 20);
            set(3, 0, 20);
            set(4 + 0, 0, score);
          });

      expect(decodeDetections(one(0.26), identityLetterbox()).length, 1);
      expect(decodeDetections(one(0.24), identityLetterbox()).length, 0);
      // Strict '>': exactly 0.25 is dropped.
      expect(decodeDetections(one(0.25), identityLetterbox()).length, 0);
    });

    test('test_coordinate_space_pixel_not_normalized keeps 928-pixel centers',
        () {
      const int anchor = 7;
      final Float32List out = buildOutput((set) {
        set(0, anchor, 464); // cx in PIXELS (==928/2), not 0.5 normalized
        set(1, anchor, 232);
        set(2, anchor, 100);
        set(3, anchor, 100);
        set(4 + 3, anchor, 0.8);
      });

      final Detection d =
          decodeDetections(out, identityLetterbox()).first;
      // Center must remain at pixel 464/232, proving no normalization.
      expect((d.left + d.right) / 2, closeTo(464, 1e-3));
      expect((d.top + d.bottom) / 2, closeTo(232, 1e-3));
    });
  });
}
