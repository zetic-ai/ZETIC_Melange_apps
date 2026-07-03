import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sayright/models/phonemes.dart';
import 'package:sayright/models/sentence.dart';
import 'package:sayright/services/ctc_aligner.dart';
import 'package:sayright/services/gop_scorer.dart';
import 'package:sayright/services/postprocessor.dart';

/// One frame per entry; the given class gets high prob, blank a low prob, rest ~0.
LogProbView lpFrames(List<int> dominantClass) {
  const c = Phonemes.classCount;
  final data = Float32List(dominantClass.length * c)
    ..fillRange(0, dominantClass.length * c, math.log(1e-6));
  for (var f = 0; f < dominantClass.length; f++) {
    data[f * c + dominantClass[f]] = math.log(0.9);
    data[f * c + Phonemes.blank] = math.log(0.05);
  }
  return LogProbView(data, frames: dominantClass.length, classes: c);
}

void main() {
  const a = 0; // AA
  const b = 6; // B
  const aligner = CtcAligner();

  group('CTC skip rule', () {
    test('DIFFERENT phonemes skip across the intervening blank (fit in 2 frames)',
        () {
      // Target [A, B], audio [A, B] with no blank frame between them.
      // The path must skip state (blank) between the two DIFFERENT phones.
      final frames = aligner.align(lpFrames([a, b]), const [a, b]);
      expect(frames[0], [0], reason: 'A owns frame 0');
      expect(frames[1], [1], reason: 'B owns frame 1 via the skip');
    });

    test('REPEATED phoneme must NOT skip: cannot fit [A,A] into 2 frames', () {
      // Skip from A to A is forbidden, so a blank frame is required between the
      // two copies. With only 2 frames one copy is squeezed out.
      final frames = aligner.align(lpFrames([a, a]), const [a, a]);
      final bothAligned = frames[0].isNotEmpty && frames[1].isNotEmpty;
      expect(bothAligned, isFalse,
          reason: 'a repeated phoneme cannot be reached by skipping');
    });

    test('REPEATED phoneme DOES fit once a blank frame separates them', () {
      // Target [A,A], audio [A, blank, A] -> both copies align.
      final frames =
          aligner.align(lpFrames([a, Phonemes.blank, a]), const [a, a]);
      expect(frames[0], [0]);
      expect(frames[1], [2]);
    });

    test('a phoneme with no aligned frames scores 0', () {
      const scorer = GopScorer();
      final sentence = PracticeSentence(
        text: 'a a',
        words: const ['a', 'a'],
        phonemeIds: const [a, a],
        spans: const [PhoneSpan(0, 1), PhoneSpan(1, 2)],
        estSeconds: 0,
      );
      // [A,A] into 2 frames -> at least one phone gets no frames.
      final words = scorer.scoreWords(lpFrames([a, a]), sentence, 0.0);
      final phones = [words[0].phonemes.single, words[1].phonemes.single];
      final empty = phones.where((p) => p.alignedFrames == 0);
      expect(empty, isNotEmpty);
      for (final p in empty) {
        expect(p.gop, 0.0);
        expect(p.score, 0.0);
      }
    });
  });
}
