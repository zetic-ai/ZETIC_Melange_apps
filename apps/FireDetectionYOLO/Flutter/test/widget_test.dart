// Unit tests for PyroGuard's pure post-processing logic (NMS + decoding).
// The camera/model pipeline needs a device, so it isn't covered here.

import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:pyroguard/models/detection.dart';
import 'package:pyroguard/services/nms.dart';
import 'package:pyroguard/services/postprocessor.dart';

void main() {
  group('IoU', () {
    test('identical boxes => 1.0', () {
      final r = const Rect.fromLTRB(0.1, 0.1, 0.5, 0.5);
      expect(iou(r, r), closeTo(1.0, 1e-9));
    });

    test('disjoint boxes => 0.0', () {
      final a = const Rect.fromLTRB(0.0, 0.0, 0.2, 0.2);
      final b = const Rect.fromLTRB(0.5, 0.5, 0.8, 0.8);
      expect(iou(a, b), 0.0);
    });

    test('half overlap', () {
      final a = const Rect.fromLTRB(0.0, 0.0, 0.2, 0.1); // area 0.02
      final b = const Rect.fromLTRB(0.1, 0.0, 0.3, 0.1); // area 0.02
      // intersection 0.1x0.1 = 0.01; union 0.03 => 1/3
      expect(iou(a, b), closeTo(1 / 3, 1e-9));
    });
  });

  group('NMS', () {
    test('suppresses overlapping lower-confidence box', () {
      final dets = [
        const Detection(
          rect: Rect.fromLTRB(0.0, 0.0, 0.2, 0.2),
          classId: kClassFire,
          confidence: 0.9,
        ),
        const Detection(
          rect: Rect.fromLTRB(0.01, 0.01, 0.21, 0.21),
          classId: kClassFire,
          confidence: 0.6,
        ),
      ];
      final kept = nonMaxSuppression(dets, 0.45);
      expect(kept.length, 1);
      expect(kept.first.confidence, 0.9);
    });

    test('keeps boxes of different classes that overlap', () {
      final dets = [
        const Detection(
          rect: Rect.fromLTRB(0.0, 0.0, 0.2, 0.2),
          classId: kClassFire,
          confidence: 0.9,
        ),
        const Detection(
          rect: Rect.fromLTRB(0.0, 0.0, 0.2, 0.2),
          classId: kClassSmoke,
          confidence: 0.8,
        ),
      ];
      final kept = nmsPerClass(dets, 0.45, classCount: 2);
      expect(kept.length, 2);
    });
  });

  group('postprocessOutput', () {
    test('decodes a single fire box (no letterbox)', () {
      // Square 640x640 source => scale 1, no padding.
      final out = Float32List(kNumChannels * kNumAnchors);
      const i = 0;
      const n = kNumAnchors;
      out[0 * n + i] = 320; // cx
      out[1 * n + i] = 320; // cy
      out[2 * n + i] = 160; // w
      out[3 * n + i] = 160; // h
      out[4 * n + i] = 0.95; // fire conf
      out[5 * n + i] = 0.10; // smoke conf

      final dets = postprocessOutput(PostprocessRequest(
        output: out,
        confThreshold: 0.25,
        scale: 1.0,
        padX: 0,
        padY: 0,
        srcWidth: 640,
        srcHeight: 640,
      ));

      expect(dets.length, 1);
      final d = dets.first;
      expect(d.classId, kClassFire);
      expect(d.confidence, closeTo(0.95, 1e-6));
      // Box center 320/640 = 0.5, extent 160/640 = 0.25 => [0.375, 0.625].
      expect(d.rect.left, closeTo(0.375, 1e-6));
      expect(d.rect.right, closeTo(0.625, 1e-6));
    });

    test('filters out sub-threshold anchors', () {
      final out = Float32List(kNumChannels * kNumAnchors);
      out[4 * kNumAnchors + 0] = 0.2; // below threshold
      final dets = postprocessOutput(PostprocessRequest(
        output: out,
        confThreshold: 0.25,
        scale: 1.0,
        padX: 0,
        padY: 0,
        srcWidth: 640,
        srcHeight: 640,
      ));
      expect(dets, isEmpty);
    });
  });
}
