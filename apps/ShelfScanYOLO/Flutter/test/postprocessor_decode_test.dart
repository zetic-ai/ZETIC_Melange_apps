import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:shelfscanyolo/services/letterbox.dart';
import 'package:shelfscanyolo/services/postprocessor.dart';

/// Identity transform: original == 640 letterbox space (scale 1, no pad), so
/// decoded box coords equal the raw 640-space coords — lets us assert the
/// decode math directly.
LetterboxTransform _identity() => LetterboxTransform.compute(
      originalWidth: 640,
      originalHeight: 640,
    );

/// Build a channel-major [1,5,8400] tensor, writing [cx,cy,w,h,score] for the
/// given anchors at their channel strides.
Float32List _buildOutput(
  Map<int, List<double>> anchors, {
  int numAnchors = 8400,
}) {
  final out = Float32List(5 * numAnchors);
  anchors.forEach((a, vals) {
    out[0 * numAnchors + a] = vals[0]; // cx
    out[1 * numAnchors + a] = vals[1]; // cy
    out[2 * numAnchors + a] = vals[2]; // w
    out[3 * numAnchors + a] = vals[3]; // h
    out[4 * numAnchors + a] = vals[4]; // score
  });
  return out;
}

void main() {
  const post = Postprocessor(); // conf 0.25, iou 0.45

  group('channel-major [1,5,8400] decode', () {
    test('reads channels by anchor stride, not 5-contiguous', () {
      // One product at anchor 100: center (320,200), size 40x60, score 0.90.
      final out = _buildOutput({
        100: [320, 200, 40, 60, 0.90],
      });
      final dets = post.decode(out, _identity());

      expect(dets, hasLength(1));
      final b = dets.single.box;
      expect(b.x1, closeTo(300, 1e-4)); // 320 - 40/2
      expect(b.y1, closeTo(170, 1e-4)); // 200 - 60/2
      expect(b.x2, closeTo(340, 1e-4)); // 320 + 40/2
      expect(b.y2, closeTo(230, 1e-4)); // 200 + 60/2
      expect(dets.single.confidence, closeTo(0.90, 1e-6));
    });

    test('a row-major writer would be silently misread -> distractor', () {
      // Write a high value at the *row-major* score slot for anchor 100
      // (index 100*5 + 4 = 504) and nothing at the true channel-major score
      // slot (4*8400 + 100). A channel-major decoder must NOT emit a detection.
      final out = Float32List(5 * 8400);
      out[100 * 5 + 4] = 0.99; // where a naive row-major reader "sees" score
      final dets = post.decode(out, _identity());
      expect(dets, isEmpty,
          reason: 'channel-major decode must ignore the row-major slot');
    });
  });

  group('score semantics (NO extra sigmoid)', () {
    test('confidence is used as-is, never re-sigmoided', () {
      // 0.30 as-is passes conf 0.25. If wrongly re-sigmoided it would become
      // sigmoid(0.30)=0.574 (still >0.25 but a WRONG value we assert against).
      final out = _buildOutput({
        7: [100, 100, 20, 20, 0.30],
      });
      final dets = post.decode(out, _identity());
      expect(dets, hasLength(1));
      // float32 stores 0.30 as ~0.30000001 — as-is, no activation applied.
      expect(dets.single.confidence, closeTo(0.30, 1e-6));
      // Guard: it is NOT the double-sigmoid value.
      const doubleSigmoid = 0.574442516; // sigmoid(0.30)
      expect((dets.single.confidence - doubleSigmoid).abs() > 0.2, isTrue);
    });
  });

  group('coordinate space is 640 pixels, not normalized 0..1', () {
    test('large raw coords survive as pixels', () {
      final out = _buildOutput({
        42: [600, 300, 20, 20, 0.8], // cx near 640 -> clearly pixel space
      });
      final dets = post.decode(out, _identity());
      expect(dets.single.box.x2, closeTo(610, 1e-4));
      expect(dets.single.box.x2, greaterThan(1.0),
          reason: 'boxes are pixels (~640), not normalized 0..1');
    });
  });

  group('threshold boundary (strict > 0.25)', () {
    test('exactly 0.25 is dropped; just above is kept', () {
      final out = _buildOutput({
        1: [100, 100, 10, 10, 0.25], // == threshold -> dropped
        2: [200, 200, 10, 10, 0.2500001], // just above -> kept
      });
      final dets = post.decode(out, _identity());
      expect(dets, hasLength(1));
      expect(dets.single.box.x1, closeTo(195, 1e-4)); // the anchor-2 box
    });

    test('0.24 dropped, 0.26 kept', () {
      final out = _buildOutput({
        3: [50, 50, 10, 10, 0.24],
        4: [60, 60, 10, 10, 0.26],
      });
      final dets = post.decode(out, _identity());
      expect(dets, hasLength(1));
      expect(dets.single.confidence, closeTo(0.26, 1e-6));
    });
  });

  test('decode rejects a wrong-sized output tensor', () {
    expect(
      () => post.decode(Float32List(100), _identity()),
      throwsArgumentError,
    );
  });
}
