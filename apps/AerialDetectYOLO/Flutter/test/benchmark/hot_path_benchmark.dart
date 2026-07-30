import 'dart:math';
import 'dart:typed_data';

import 'package:aerialdetect/services/postprocessor.dart';
import 'package:aerialdetect/services/preprocessor.dart';
import 'package:flutter_test/flutter_test.dart';

/// A4 micro-benchmark: the full pure-Dart hot path (preprocess + decode + NMS)
/// over mock tensors of the REAL shape ([1,3,928,928] in, [1,14,17661] out).
/// This is the only honest latency number an agent can produce — it excludes
/// the device-only model.run(). Reports median + p90 over many iterations.
///
/// Run: flutter test test/benchmark/hot_path_benchmark.dart
void main() {
  test('A4 hot-path micro-benchmark (928 preprocess + 17661 decode + NMS)', () {
    const int srcW = 720, srcH = 1280; // realistic upright camera buffer
    const int bytesPerRow = srcW * 4;
    const int warmIters = 20;
    const int iters = 120;

    // --- Mock BGRA input (deterministic noise). ---
    final Uint8List bgra = Uint8List(bytesPerRow * srcH);
    final Random rng = Random(42);
    for (int i = 0; i < bgra.length; i++) {
      bgra[i] = rng.nextInt(256);
    }

    // --- Mock output: mostly low, with ~60 above-threshold anchors so NMS and
    //     box geometry actually run (a realistic detection load). ---
    final Float32List output = Float32List(kNumChannels * kNumAnchors);
    for (int a = 0; a < kNumAnchors; a++) {
      // Low baseline class scores (below 0.25) for every anchor.
      for (int c = 0; c < kNumClassChannels; c++) {
        output[(4 + c) * kNumAnchors + a] = rng.nextDouble() * 0.15;
      }
    }
    for (int k = 0; k < 60; k++) {
      final int a = rng.nextInt(kNumAnchors);
      final int cls = rng.nextInt(kNumClassChannels);
      output[a] = rng.nextDouble() * 928; // cx
      output[kNumAnchors + a] = rng.nextDouble() * 928; // cy
      output[2 * kNumAnchors + a] = 40 + rng.nextDouble() * 80; // w
      output[3 * kNumAnchors + a] = 40 + rng.nextDouble() * 80; // h
      output[(4 + cls) * kNumAnchors + a] = 0.3 + rng.nextDouble() * 0.6;
    }

    final Float32List inputBuffer = Float32List(kInputElements);
    final LetterboxParams lb = computeLetterbox(srcW, srcH);

    void hotPath() {
      letterboxBgraToNchw(bgra, srcW, srcH, bytesPerRow, lb, inputBuffer);
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

    // Sanity: the hot path completes and produces a finite number.
    expect(median, greaterThan(0));
  });
}
