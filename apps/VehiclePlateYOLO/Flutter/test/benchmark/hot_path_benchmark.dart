import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vehicleplateyolo/services/frame_data.dart';
import 'package:vehicleplateyolo/services/postprocessor.dart';
import 'package:vehicleplateyolo/services/preprocessor.dart';

/// A4 micro-benchmark: feeds mock tensors of the REAL shapes through the full
/// pure-Dart hot path (preprocess + decode + NMS) and reports median/p90 over
/// many iterations. Excludes model.run (NPU, device-only). The median is the
/// Tier B post-processing budget baseline.
///
/// Run: flutter test test/benchmark/hot_path_benchmark.dart
void main() {
  test('hot path median/p90 over many iterations', () {
    const iterations = 300;
    const warmup = 30;
    const n = 8400;

    // Mock iOS BGRA frame (720x1280 upright), deterministic pseudo-random bytes.
    final rnd = math.Random(42);
    final bgra = Uint8List(720 * 1280 * 4);
    for (var i = 0; i < bgra.length; i++) {
      bgra[i] = rnd.nextInt(256);
    }
    final frame = FrameData(
      format: FramePixelFormat.bgra8888,
      width: 720,
      height: 1280,
      rotationDegrees: 0,
      plane0: bgra,
      bytesPerRow0: 720 * 4,
    );

    // Mock model output [1,5,8400] with a realistic candidate load (~250 anchors
    // above threshold, clustered so NMS has real work).
    final output = Float32List(5 * n);
    for (var a = 0; a < n; a++) {
      output[4 * n + a] = 0.05 + rnd.nextDouble() * 0.15; // mostly below 0.25
    }
    for (var k = 0; k < 250; k++) {
      final a = rnd.nextInt(n);
      output[0 * n + a] = rnd.nextDouble() * 640;
      output[1 * n + a] = rnd.nextDouble() * 640;
      output[2 * n + a] = 20 + rnd.nextDouble() * 80;
      output[3 * n + a] = 20 + rnd.nextDouble() * 40;
      output[4 * n + a] = 0.3 + rnd.nextDouble() * 0.6; // above threshold
    }

    final pre = Preprocessor();
    const post = Postprocessor();

    final samples = <int>[];
    for (var i = 0; i < iterations + warmup; i++) {
      final sw = Stopwatch()..start();
      final r = pre.process(frame);
      final dets = post.decode(output, r.params);
      sw.stop();
      if (i >= warmup) samples.add(sw.elapsedMicroseconds);
      // Keep the optimizer honest (use the result).
      expect(dets, isNotNull);
    }

    samples.sort();
    final median = samples[samples.length ~/ 2] / 1000.0;
    final p90 = samples[(samples.length * 0.9).floor()] / 1000.0;
    final mean =
        samples.reduce((a, b) => a + b) / samples.length / 1000.0;

    // ignore: avoid_print
    print(
      'A4 hot-path (preprocess+decode+NMS) over $iterations iters: '
      'median=${median.toStringAsFixed(2)}ms '
      'p90=${p90.toStringAsFixed(2)}ms '
      'mean=${mean.toStringAsFixed(2)}ms',
    );

    expect(median, greaterThan(0));
  });
}
