import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:retinadrgrade/services/postprocessor.dart';
import 'package:retinadrgrade/services/preprocessor.dart';

/// A4 — pure-Dart hot-path micro-benchmark (VALIDATION.md).
///
/// Feeds a mock fundus-sized image through the FULL pure-Dart hot path
/// (decode + PLAIN resize-224 + normalize + NCHW + softmax + argmax) and reports
/// the median over many iterations. This is the post-processing budget an agent
/// can measure honestly; it is NOT the end-to-end device latency (the NPU/CPU
/// inference time is fixed by Melange and only appears on hardware — EXPECTED
/// NPU ~10 ms / CPU ~598 ms; GPU is a crash path).
void main() {
  test('hot-path median latency (preprocess + softmax + argmax)', () {
    const post = Postprocessor();

    // Build a mock fundus-sized RGB image once and encode to PNG bytes, so each
    // iteration exercises the same decode->preprocess path the app runs.
    final rand = math.Random(42);
    final mock = img.Image(width: 640, height: 480, numChannels: 3);
    for (final p in mock) {
      p
        ..r = rand.nextInt(256)
        ..g = rand.nextInt(256)
        ..b = rand.nextInt(256);
    }
    final Uint8List bytes = img.encodePng(mock);
    final mockLogits = [1.2, -0.4, 0.3, 2.1, -1.0];

    const warmup = 5;
    const iterations = 50;

    for (var i = 0; i < warmup; i++) {
      Preprocessor.preprocess(bytes);
      post.classify(mockLogits);
    }

    final samples = <int>[];
    for (var i = 0; i < iterations; i++) {
      final watch = Stopwatch()..start();
      final input = Preprocessor.preprocess(bytes);
      post.classify(mockLogits);
      watch.stop();
      // Touch the result so nothing is optimized away.
      expect(input.length, Preprocessor.tensorLength);
      samples.add(watch.elapsedMicroseconds);
    }

    samples.sort();
    final medianUs = samples[samples.length ~/ 2];
    final minUs = samples.first;
    final maxUs = samples.last;

    // ignore: avoid_print
    print(
      'A4 hot-path (640x480 mock, $iterations iters): '
      'median ${(medianUs / 1000).toStringAsFixed(2)} ms '
      '(min ${(minUs / 1000).toStringAsFixed(2)} ms, '
      'max ${(maxUs / 1000).toStringAsFixed(2)} ms)',
    );

    expect(samples.length, iterations);
  });
}
