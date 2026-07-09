import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sayright/models/sentence.dart';
import 'package:sayright/services/gop_scorer.dart';
import 'package:sayright/services/postprocessor.dart';

/// Build a LogProbView from per-frame probability rows (each row should sum to 1).
LogProbView lpFromProbs(List<List<double>> probs) {
  final t = probs.length;
  final c = probs[0].length;
  final data = Float32List(t * c);
  for (var f = 0; f < t; f++) {
    for (var cc = 0; cc < c; cc++) {
      data[f * c + cc] = math.log(probs[f][cc]);
    }
  }
  return LogProbView(data, frames: t, classes: c);
}

void main() {
  group('logprob semantics', () {
    test('outputs are log-probabilities: exp of each frame sums to ~1', () {
      // A few valid distributions over 45 classes.
      final rows = <List<double>>[];
      for (var f = 0; f < 5; f++) {
        final row = List<double>.filled(45, 0.0);
        // A crude but valid distribution.
        var sum = 0.0;
        for (var c = 0; c < 45; c++) {
          row[c] = 1.0 + (c + f) % 7;
          sum += row[c];
        }
        for (var c = 0; c < 45; c++) {
          row[c] /= sum;
        }
        rows.add(row);
      }
      final lp = lpFromProbs(rows);
      for (var f = 0; f < rows.length; f++) {
        var s = 0.0;
        for (var c = 0; c < 45; c++) {
          s += math.exp(lp.at(f, c));
        }
        expect(s, closeTo(1.0, 1e-6), reason: 'frame $f exp-sum');
      }
    });

    test('GOP of a phone whose posterior is p equals p', () {
      const targetId = 5; // AY
      const p = 0.6;
      // Every frame: posterior(targetId)=0.6, class0=0.3, blank=0.1 -> sums to 1.
      final rows = List.generate(6, (_) {
        final row = List<double>.filled(45, 0.0);
        row[targetId] = p;
        row[0] = 0.3;
        row[44] = 0.1;
        return row;
      });
      final lp = lpFromProbs(rows);
      final sentence = PracticeSentence(
        text: 'x',
        words: const ['x'],
        phonemeIds: const [targetId],
        spans: const [PhoneSpan(0, 1)],
        estSeconds: 0,
      );
      const scorer = GopScorer();
      // Full window fill -> no calibration boost for this check; assert RAW gop.
      final words = scorer.scoreWords(lp, sentence, 0.0);
      final phone = words.single.phonemes.single;
      expect(phone.alignedFrames, rows.length,
          reason: 'phone should own every frame');
      expect(phone.gop, closeTo(p, 1e-6));
    });
  });
}
