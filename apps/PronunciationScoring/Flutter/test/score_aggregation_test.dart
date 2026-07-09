import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sayright/models/phonemes.dart';
import 'package:sayright/models/sentence.dart';
import 'package:sayright/services/gop_scorer.dart';
import 'package:sayright/services/postprocessor.dart';

/// Build frames where frame f has posterior [probs[f]] on class [ids[f]] and a
/// small blank probability, everything else ~0.
LogProbView build(List<int> ids, List<double> probs) {
  const c = Phonemes.classCount;
  final data = Float32List(ids.length * c)
    ..fillRange(0, ids.length * c, math.log(1e-6));
  for (var f = 0; f < ids.length; f++) {
    data[f * c + ids[f]] = math.log(probs[f]);
    data[f * c + Phonemes.blank] = math.log(0.02);
  }
  return LogProbView(data, frames: ids.length, classes: c);
}

void main() {
  const a = 0, b = 6, cc = 7; // AA, B, CH

  // Two words: "ab" = [A,B], "c" = [C]. One frame per phone (skips allowed
  // across the different phonemes), GOPs 0.9 / 0.6 / 0.3.
  final sentence = PracticeSentence(
    text: 'ab c',
    words: const ['ab', 'c'],
    phonemeIds: const [a, b, cc],
    spans: const [PhoneSpan(0, 2), PhoneSpan(2, 3)],
    estSeconds: 0,
  );

  const scorer = GopScorer();

  group('score aggregation', () {
    test('raw per-phoneme GOPs come from the aligned posteriors', () {
      final lp = build([a, b, cc], [0.9, 0.6, 0.3]);
      final words = scorer.scoreWords(lp, sentence, 0.0);
      // float32 storage -> compare at 1e-6.
      expect(words[0].phonemes[0].gop, closeTo(0.9, 1e-6));
      expect(words[0].phonemes[1].gop, closeTo(0.6, 1e-6));
      expect(words[1].phonemes[0].gop, closeTo(0.3, 1e-6));
    });

    test('word score is the mean of its phoneme scores', () {
      final lp = build([a, b, cc], [0.9, 0.6, 0.3]);
      final words = scorer.scoreWords(lp, sentence, 0.0);
      // full-fill calibration (bf=0): expGood 0.75 -> 0.9->100(clamp), 0.6->80, 0.3->40
      expect(words[0].phonemes[0].score, closeTo(100.0, 1e-6));
      expect(words[0].phonemes[1].score, closeTo(80.0, 1e-6));
      expect(words[0].score, closeTo(90.0, 1e-6)); // mean(100, 80)
      expect(words[1].score, closeTo(40.0, 1e-6));
    });

    test('weakest phoneme is highlighted per word', () {
      final lp = build([a, b, cc], [0.9, 0.6, 0.3]);
      final words = scorer.scoreWords(lp, sentence, 0.0);
      expect(words[0].weakestPhonemeIndex, 1); // B (80) < A (100)
      expect(words[0].weakestPhoneme!.phoneme, 'B');
      expect(words[1].weakestPhonemeIndex, 0); // single phone
    });

    test('overall is the mean of word scores', () {
      final lp = build([a, b, cc], [0.9, 0.6, 0.3]);
      final words = scorer.scoreWords(lp, sentence, 0.0);
      expect(scorer.overall(words), closeTo(65.0, 1e-6)); // mean(90, 40)
    });

    test('empty word list -> overall 0', () {
      expect(scorer.overall(const []), 0.0);
    });
  });
}
