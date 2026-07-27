import 'dart:math' as math;
import 'dart:typed_data';

import 'package:contentmoderation/services/postprocessor.dart';
import 'package:contentmoderation/services/preprocessor.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

/// A4 — pure-Dart hot-path micro-benchmark (VALIDATION.md).
///
/// Feeds a realistically-sized mock image through the FULL pure-Dart hot path
/// (decode + antialiased-bicubic resize + center-crop + normalize + NCHW +
/// softmax + banding) and reports the median over many iterations. This is the
/// post-processing budget an agent can measure honestly; it is NOT the
/// end-to-end device latency (the NPU/CPU inference time is fixed by Melange and
/// only appears on hardware).
void main() {
  test('hot-path median latency (preprocess + softmax + band)', () {
    const post = Postprocessor();

    // A 1440x1080 mock photo (a typical phone-camera aspect), encoded to JPEG so
    // each iteration exercises the same decode->preprocess path the app runs.
    final rand = math.Random(42);
    final mock = img.Image(width: 1440, height: 1080, numChannels: 3);
    for (final p in mock) {
      p
        ..r = rand.nextInt(256)
        ..g = rand.nextInt(256)
        ..b = rand.nextInt(256);
    }
    final Uint8List bytes = img.encodeJpg(mock, quality: 90);
    final mockLogits = [1.2, -0.4];

    const warmup = 3;
    const iterations = 30;

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
      expect(input.length, Preprocessor.tensorLength);
      samples.add(watch.elapsedMicroseconds);
    }

    samples.sort();
    final medianUs = samples[samples.length ~/ 2];
    final minUs = samples.first;
    final maxUs = samples.last;

    // ignore: avoid_print
    print(
      'A4 hot-path (1440x1080 mock, $iterations iters): '
      'median ${(medianUs / 1000).toStringAsFixed(2)} ms '
      '(min ${(minUs / 1000).toStringAsFixed(2)} ms, '
      'max ${(maxUs / 1000).toStringAsFixed(2)} ms)',
    );

    expect(samples.length, iterations);
  });
}
