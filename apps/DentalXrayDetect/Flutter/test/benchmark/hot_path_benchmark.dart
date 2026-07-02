import 'dart:math';
import 'dart:typed_data';

import 'package:dentalxraydetect/services/postprocessor.dart';
import 'package:dentalxraydetect/services/preprocessor.dart';
import 'package:flutter_test/flutter_test.dart';

/// A4 micro-benchmark: the full pure-Dart hot path (preprocess + decode + NMS)
/// over mock tensors of the REAL shape (packed RGB in, [1,7,8400] out). This is
/// the only honest latency number an agent can produce — it EXCLUDES the
/// device-only model.run(). Reports median + p90 over many iterations.
///
/// Run: flutter test test/benchmark/hot_path_benchmark.dart
void main() {
  test('A4 hot-path micro-benchmark (640 preprocess + 8400 decode + NMS)', () {
    // A realistic (downscaled) panoramic radiograph: wide and short.
    const int srcW = 2048, srcH = 1072;
    const int warmIters = 20;
    const int iters = 120;

    // --- Mock packed RGB input (deterministic noise). ---
    final Uint8List rgb = Uint8List(srcW * srcH * 3);
    final Random rng = Random(42);
    for (int i = 0; i < rgb.length; i++) {
      rgb[i] = rng.nextInt(256);
    }

    // --- Mock output: mostly low, with ~40 above-threshold anchors so NMS and
    //     box geometry actually run (a realistic dental detection load). ---
    final Float32List output = Float32List(kNumChannels * kNumAnchors);
    for (int a = 0; a < kNumAnchors; a++) {
      for (int c = 0; c < kNumClassChannels; c++) {
        output[(4 + c) * kNumAnchors + a] = rng.nextDouble() * 0.2;
      }
    }
    for (int k = 0; k < 40; k++) {
      final int a = rng.nextInt(kNumAnchors);
      final int cls = rng.nextInt(kNumClassChannels);
      output[a] = rng.nextDouble() * 640; // cx
      output[kNumAnchors + a] = rng.nextDouble() * 640; // cy
      output[2 * kNumAnchors + a] = 30 + rng.nextDouble() * 60; // w
      output[3 * kNumAnchors + a] = 30 + rng.nextDouble() * 60; // h
      output[(4 + cls) * kNumAnchors + a] = 0.46 + rng.nextDouble() * 0.5;
    }

    final Float32List inputBuffer = Float32List(kInputElements);
    final LetterboxParams lb = computeLetterbox(srcW, srcH);

    void hotPath() {
      letterboxRgbToNchw(rgb, srcW, srcH, lb, inputBuffer);
      decodeDetections(output, lb);
    }

    for (int i = 0; i < warmIters; i++) {
      hotPath();
    }

    final List<double> samples = <double>[];
    for (int i = 0; i < iters; i++) {
      final Stopwatch sw = Stopwatch()..start();
      hotPath();
      sw.stop();
      samples.add(sw.elapsedMicroseconds / 1000.0);
    }
    samples.sort();
    final double median = samples[samples.length ~/ 2];
    final double p90 = samples[(samples.length * 0.9).floor()];
    final double minMs = samples.first;
    final double maxMs = samples.last;

    // ignore: avoid_print
    print('A4 hot-path (pre+decode+NMS) over $iters iters: '
        'median=${median.toStringAsFixed(2)}ms '
        'p90=${p90.toStringAsFixed(2)}ms '
        'min=${minMs.toStringAsFixed(2)}ms '
        'max=${maxMs.toStringAsFixed(2)}ms');

    expect(median, greaterThan(0));
  });
}
