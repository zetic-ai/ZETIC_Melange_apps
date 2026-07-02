import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:shelfscanyolo/services/letterbox.dart';
import 'package:shelfscanyolo/services/nms.dart';
import 'package:shelfscanyolo/services/postprocessor.dart';
import 'package:shelfscanyolo/services/preprocessor.dart';

/// Pure-Dart hot-path micro-benchmark (VALIDATION.md A4): feeds a realistic
/// decoded image + a synthetic [1,5,8400] output through preprocess + decode +
/// NMS and reports the median wall time. This is the post-processing budget an
/// agent can measure honestly — NOT end-to-end device latency (the NPU time is
/// only visible on hardware).
///
/// Run: flutter test test/benchmark/hot_path_benchmark.dart
void main() {
  test('hot path median latency (preprocess + decode + NMS)', () {
    const iterations = 40;
    const warmup = 5;

    // Realistic decoded frame (phone-sized landscape shelf photo).
    final image = _syntheticImage(1440, 1080);

    // Synthetic dense output: ~2000 above-threshold anchors that overlap, so
    // NMS does real O(n^2) work like a packed shelf.
    final output = _syntheticOutput(aboveThreshold: 2000);
    final transform =
        LetterboxTransform.compute(originalWidth: 1440, originalHeight: 1080);
    const pre = Preprocessor();
    const post = Postprocessor();

    final samples = <double>[];
    for (var i = 0; i < iterations + warmup; i++) {
      final sw = Stopwatch()..start();
      final p = pre.processDecoded(image);
      final raw = post.decode(output, transform);
      final kept = nonMaxSuppression(raw, post.iouThreshold);
      sw.stop();
      // Touch results so nothing is optimized away.
      if (p.input.isEmpty || kept.length > raw.length) {
        fail('unreachable');
      }
      if (i >= warmup) samples.add(sw.elapsedMicroseconds / 1000.0);
    }

    samples.sort();
    final median = samples[samples.length ~/ 2];
    final p90 = samples[(samples.length * 0.9).floor().clamp(0, samples.length - 1)];
    // ignore: avoid_print
    print('HOT-PATH BENCHMARK (n=$iterations): '
        'median=${median.toStringAsFixed(2)}ms '
        'min=${samples.first.toStringAsFixed(2)}ms '
        'p90=${p90.toStringAsFixed(2)}ms');

    // Generous ceiling so the test is a report, not a flaky gate.
    expect(median, lessThan(3000));
  });
}

/// A deterministic noise image (no file IO) at the given size.
img.Image _syntheticImage(int w, int h) {
  final image = img.Image(width: w, height: h, numChannels: 3);
  final rnd = math.Random(7);
  for (final p in image) {
    p
      ..r = rnd.nextInt(256)
      ..g = rnd.nextInt(256)
      ..b = rnd.nextInt(256);
  }
  return image;
}

/// Channel-major [1,5,8400] with [aboveThreshold] overlapping anchors above the
/// 0.25 confidence threshold, the rest below.
Float32List _syntheticOutput({required int aboveThreshold, int numAnchors = 8400}) {
  final out = Float32List(5 * numAnchors);
  final rnd = math.Random(11);
  for (var a = 0; a < numAnchors; a++) {
    final hot = a < aboveThreshold;
    final cx = 20.0 + rnd.nextDouble() * 600;
    final cy = 20.0 + rnd.nextDouble() * 600;
    out[0 * numAnchors + a] = cx;
    out[1 * numAnchors + a] = cy;
    out[2 * numAnchors + a] = 24.0; // overlapping-ish boxes
    out[3 * numAnchors + a] = 24.0;
    out[4 * numAnchors + a] = hot ? 0.3 + rnd.nextDouble() * 0.6 : 0.05;
  }
  return out;
}
