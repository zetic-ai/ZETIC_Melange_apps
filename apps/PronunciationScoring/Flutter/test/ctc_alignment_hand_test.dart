import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sayright/models/phonemes.dart';
import 'package:sayright/models/sentence.dart';
import 'package:sayright/services/ctc_aligner.dart';
import 'package:sayright/services/gop_scorer.dart';
import 'package:sayright/services/postprocessor.dart';

// Hand-worked 3-frame example, target = [AA(0)], lattice ext = [blank, AA, blank].
// Posteriors (only AA and blank matter — the aligner reads no other class):
//   f0: P(blank)=0.8  P(AA)=0.2
//   f1: P(blank)=0.1  P(AA)=0.9
//   f2: P(blank)=0.7  P(AA)=0.3
// Worked Viterbi best path = [blank, AA, blank] -> AA owns ONLY frame 1,
// so GOP(AA) = P(AA @ f1) = 0.9. (See test comment block in the file for the
// full DP grid.)
LogProbView buildHandLp() {
  const c = Phonemes.classCount;
  const t = 3;
  final data = Float32List(t * c)..fillRange(0, t * c, -50.0);
  void set(int f, int cls, double p) => data[f * c + cls] = math.log(p);
  set(0, 0, 0.2);
  set(0, 44, 0.8);
  set(1, 0, 0.9);
  set(1, 44, 0.1);
  set(2, 0, 0.3);
  set(2, 44, 0.7);
  return LogProbView(data, frames: t, classes: c);
}

void main() {
  group('CTC forced alignment (hand-computed grid)', () {
    test('single phone aligns to exactly frame 1', () {
      const aligner = CtcAligner();
      final frames = aligner.align(buildHandLp(), const [0]);
      expect(frames.length, 1);
      expect(frames[0], [1]);
    });

    test('GOP equals the posterior on the single aligned frame (0.9)', () {
      const scorer = GopScorer();
      final sentence = PracticeSentence(
        text: 'a',
        words: const ['a'],
        phonemeIds: const [0],
        spans: const [PhoneSpan(0, 1)],
        estSeconds: 0,
      );
      final words = scorer.scoreWords(buildHandLp(), sentence, 0.0);
      expect(words.single.phonemes.single.gop, closeTo(0.9, 1e-9));
    });
  });
}
