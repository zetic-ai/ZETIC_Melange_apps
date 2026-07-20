import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:voxscribe/services/detokenizer.dart';
import 'package:voxscribe/services/log_mel.dart';
import 'package:voxscribe/services/postprocessor.dart';
import 'package:voxscribe/services/preprocessor.dart';

/// A4 — Hot-path micro-benchmark (VALIDATION.md A4).
///
/// Feeds mock tensors of the REAL shapes through the full pure-Dart hot path,
/// mocking the 3 native run() calls so only Dart cost is measured:
///   * log-mel: 480000-sample span -> [1,80,3000]  (heaviest Dart stage);
///   * powerset decode + onset/offset over a mock [1,589,7];
///   * greedy decode argmax over 51865 logits x ~50 steps from a mock
///     [1,448,51865] buffer (row idx-1 each step);
///   * detokenization of ~50 ids.
/// Reports the median over N iterations — the Tier B baseline (post-processing
/// budget, NOT end-to-end device latency).
void main() {
  test('hot-path median (log-mel + powerset + greedy-argmax + detok)', () {
    // --- filterbank for log-mel ---
    final Uint8List fb = File('assets/mel_filters_80.bin').readAsBytesSync();
    final ByteData bd = ByteData.sublistView(fb);
    final Float32List filters = Float32List(fb.length ~/ 4);
    for (int i = 0; i < filters.length; i++) {
      filters[i] = bd.getFloat32(i * 4, Endian.little);
    }
    final LogMel logMel = LogMel(filters);

    // --- mock span (2 s of speech-like signal, padded to 30 s) ---
    final Float32List span = Float32List(kWhisperSpanSamples);
    for (int i = 0; i < 32000; i++) {
      span[i] = 0.4 * math.sin(2 * math.pi * 300.0 * i / 16000.0);
    }

    // --- mock segmentation logits [1,589,7] ---
    final math.Random rng = math.Random(7);
    final Float32List segLogits = Float32List(kSegFrames * kSegClasses);
    for (int i = 0; i < segLogits.length; i++) {
      segLogits[i] = rng.nextDouble();
    }

    // --- mock decoder logits [1,448,51865]; each row argmaxes to a distinct
    //     token so the repetition guard never fires (full ~50-step run) ---
    const int steps = 50;
    final Float32List decLogits = Float32List(kMaxLen * kVocab);
    for (int r = 0; r <= steps; r++) {
      decLogits[r * kVocab + (r * 977 % 50000)] = 10.0;
    }
    Float32List step(Int32List ids, Int32List mask) => decLogits;

    // Detokenizer over the produced ids (~50 ids), per the A4 stage list.
    final Detokenizer detok = Detokenizer.fromVocabJson(
        File('assets/vocab.json').readAsStringSync());

    int hot() {
      final LogMelResult mel = logMel.compute(span);
      final List<List<bool>> labels = powersetDecode(segLogits);
      final List<dynamic> segs = onsetOffsetSegments(labels);
      final List<int> ids =
          greedyDecode(step, maxLength: steps + 1, repetitionGuard: 0);
      final String text = detok.decode(ids);
      // touch outputs so nothing is optimized away
      return mel.frames + labels.length + segs.length + ids.length + text.length;
    }

    // Warm-up.
    for (int i = 0; i < 2; i++) {
      hot();
    }

    const int iters = 9;
    final List<double> samples = List<double>.filled(iters, 0);
    for (int i = 0; i < iters; i++) {
      final Stopwatch sw = Stopwatch()..start();
      hot();
      sw.stop();
      samples[i] = sw.elapsedMicroseconds / 1000.0;
    }
    samples.sort();
    final double median = samples[iters ~/ 2];
    final double p90 = samples[(iters * 0.9).floor()];

    // ignore: avoid_print
    print('A4 HOT-PATH  median=${median.toStringAsFixed(1)}ms  '
        'p90=${p90.toStringAsFixed(1)}ms  (log-mel+powerset+greedy+detok, '
        '${iters}x)');

    expect(median, greaterThan(0));
  }, timeout: const Timeout(Duration(minutes: 5)));
}
