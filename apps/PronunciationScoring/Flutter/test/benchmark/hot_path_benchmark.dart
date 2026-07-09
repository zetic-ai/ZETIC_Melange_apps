import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sayright/models/phonemes.dart';
import 'package:sayright/models/sentence.dart';
import 'package:sayright/services/gop_scorer.dart';
import 'package:sayright/services/postprocessor.dart';

/// Micro-benchmark of the pure-Dart hot path (greedy decode + CTC forced
/// alignment + GOP scoring) on a mock [64,45] logprob tensor. This is the
/// scoring head that runs once per recording on the main isolate.
///
/// Tier-B baseline: run `flutter test test/benchmark/hot_path_benchmark.dart`
/// and read the printed median. There is exactly ONE inference per 5.11 s
/// recording, so this path is not latency-critical, but the median is recorded
/// as the optimization baseline.
void main() {
  test('hot path median over >=1000 iterations', () {
    final rng = math.Random(42);

    // Mock log-softmax: random logits normalized to a valid log-distribution.
    final data = Float32List(Phonemes.frameCount * Phonemes.classCount);
    for (var f = 0; f < Phonemes.frameCount; f++) {
      final logits = List<double>.generate(
          Phonemes.classCount, (_) => rng.nextDouble() * 6 - 3);
      final maxL = logits.reduce(math.max);
      var sumExp = 0.0;
      for (final l in logits) {
        sumExp += math.exp(l - maxL);
      }
      final logZ = maxL + math.log(sumExp);
      for (var c = 0; c < Phonemes.classCount; c++) {
        data[f * Phonemes.classCount + c] = logits[c] - logZ;
      }
    }
    final lp = LogProbView(data);

    // A ~40-phone target across ~10 words (representative sentence length).
    final ids = List<int>.generate(40, (i) => (i * 13 + 3) % 39);
    final spans = <PhoneSpan>[];
    for (var w = 0; w < 10; w++) {
      spans.add(PhoneSpan(w * 4, w * 4 + 4));
    }
    final sentence = PracticeSentence(
      text: 'benchmark',
      words: List<String>.generate(10, (i) => 'w$i'),
      phonemeIds: ids,
      spans: spans,
      estSeconds: 4.2,
    );

    const scorer = GopScorer();

    void once() {
      final greedy = greedyDecode(lp);
      final words = scorer.scoreWords(lp, sentence, greedy.blankFraction);
      scorer.overall(words);
    }

    // Warm up (JIT).
    for (var i = 0; i < 200; i++) {
      once();
    }

    const iters = 2000;
    final samples = Float64List(iters);
    for (var i = 0; i < iters; i++) {
      final sw = Stopwatch()..start();
      once();
      sw.stop();
      samples[i] = sw.elapsedMicroseconds.toDouble();
    }
    final sorted = samples.toList()..sort();
    final medianUs = sorted[iters ~/ 2];
    final p95Us = sorted[(iters * 95) ~/ 100];

    // ignore: avoid_print
    print('HOT PATH decode+align+score over $iters iters: '
        'median=${medianUs.toStringAsFixed(1)}us '
        'p95=${p95Us.toStringAsFixed(1)}us');

    // Generous sanity bound (this path is sub-millisecond on desktop).
    expect(medianUs, lessThan(5000),
        reason: 'scoring head should stay well under 5 ms');
  });
}
