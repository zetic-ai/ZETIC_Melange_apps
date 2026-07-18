// Hot-path micro-benchmark (VALIDATION.md A4 / Tier B).
//
// Feeds mock frames of the real shape through the full pure-Dart hot path —
// frame -> upright BGR -> PAD 2.7x crop+resize -> ArcFace warp -> softmax ->
// L2-norm -> cosine — and reports the median over many iterations. This is the
// post-processing budget the worker controls; it is NOT end-to-end device
// latency (the NPU time is fixed by Melange and only appears on hardware).
//
// Run: flutter test test/benchmark/hot_path_benchmark.dart
//
// Tier B compares two variants on this same harness:
//   * baseline : allocate the input Float32Lists fresh every frame.
//   * optimized: reuse pre-allocated input buffers (buildPadInput/buildFaceInput
//     `out:` argument) across frames.

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:livegate/models/geometry.dart';
import 'package:livegate/services/face_align.dart';
import 'package:livegate/services/frame.dart';
import 'package:livegate/services/gate.dart';
import 'package:livegate/services/preprocessor.dart';

FrameData mockFrame(int w, int h) {
  final bytes = Uint8List(w * h * 4);
  for (var i = 0; i < bytes.length; i++) {
    bytes[i] = (i * 31 + 7) & 0xFF;
  }
  return FrameData.bgra8888(
    width: w,
    height: h,
    bgra: bytes,
    bgraRowStride: w * 4,
    rotationDegrees: 0,
  );
}

const _box = FaceBox(x: 250, y: 300, w: 220, h: 260);
const _lm = Landmarks5(
  leftEye: Point2(320, 380),
  rightEye: Point2(430, 380),
  nose: Point2(375, 440),
  mouthLeft: Point2(335, 500),
  mouthRight: Point2(415, 500),
);

int _median(List<int> xs) {
  xs.sort();
  return xs[xs.length ~/ 2];
}

/// Runs the pure-Dart hot path once. When [padOut]/[faceOut] are supplied they
/// are reused (the optimized variant); otherwise fresh buffers are allocated.
void hotPath(FrameData frame, Float32List enrolled,
    {Float32List? padOut, Float32List? faceOut}) {
  final img = frameToUprightBgr(frame);
  final crop = computeCropBox(img.width, img.height, _box);
  final padInput = buildPadInput(img, crop, out: padOut);
  final faceInput = buildFaceInput(img, _lm, out: faceOut);

  // Mock model outputs (the NPU runs on-device; here we exercise the decode).
  final padLogits = [padInput[0] * 0.001, padInput[1] * 0.01, padInput[2] * 0.001];
  final live = livenessScore(padLogits);
  if (live >= 0) {
    final probe = l2normalize(Float32List.sublistView(faceInput, 0, 128));
    cosine(probe, enrolled);
  }
}

void main() {
  test('hot-path micro-benchmark: baseline vs pre-allocated buffers', () {
    const w = 720, h = 1280; // realistic front-camera buffer
    final frame = mockFrame(w, h);
    final enrolled = l2normalize(
        Float32List.fromList(List<double>.generate(128, (i) => (i % 7) - 3.0)));

    const warm = 20;
    const iters = 120;

    // Warm up (JIT).
    for (var i = 0; i < warm; i++) {
      hotPath(frame, enrolled);
    }

    // Baseline: fresh allocations every frame.
    final baseline = <int>[];
    for (var i = 0; i < iters; i++) {
      final sw = Stopwatch()..start();
      hotPath(frame, enrolled);
      sw.stop();
      baseline.add(sw.elapsedMicroseconds);
    }

    // Optimized: reuse pre-allocated input buffers.
    final padOut = Float32List(3 * kPadInputSize * kPadInputSize);
    final faceOut = Float32List(3 * kFaceInputSize * kFaceInputSize);
    for (var i = 0; i < warm; i++) {
      hotPath(frame, enrolled, padOut: padOut, faceOut: faceOut);
    }
    final optimized = <int>[];
    for (var i = 0; i < iters; i++) {
      final sw = Stopwatch()..start();
      hotPath(frame, enrolled, padOut: padOut, faceOut: faceOut);
      sw.stop();
      optimized.add(sw.elapsedMicroseconds);
    }

    final b = _median(baseline);
    final o = _median(optimized);
    final delta = (b - o) / b * 100;
    // ignore: avoid_print
    print('hot-path median: baseline=${b}us  optimized=${o}us  '
        'delta=${delta.toStringAsFixed(1)}%');

    expect(b, greaterThan(0));
    expect(o, greaterThan(0));
  });
}
