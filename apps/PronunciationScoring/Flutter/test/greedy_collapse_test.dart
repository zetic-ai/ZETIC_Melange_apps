import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sayright/models/phonemes.dart';
import 'package:sayright/services/postprocessor.dart';

/// LogProbView whose per-frame argmax is exactly [argmaxIds].
LogProbView lpFromArgmax(List<int> argmaxIds) {
  const c = Phonemes.classCount;
  final data = Float32List(argmaxIds.length * c);
  for (var f = 0; f < argmaxIds.length; f++) {
    for (var cc = 0; cc < c; cc++) {
      data[f * c + cc] = -5.0;
    }
    data[f * c + argmaxIds[f]] = 5.0;
  }
  return LogProbView(data, frames: argmaxIds.length, classes: c);
}

void main() {
  const l = 20; // L
  const k = 19; // K
  const blank = Phonemes.blank; // 44
  const special = 40; // [PAD]-special, must be dropped

  group('greedy CTC collapse semantics', () {
    test('"L <blank> L" -> [L, L] (blank breaks the run)', () {
      final out = greedyDecode(lpFromArgmax([l, blank, l])).phonemes;
      expect(out, ['L', 'L']);
    });

    test('"L L" -> [L] (adjacent repeat collapses)', () {
      final out = greedyDecode(lpFromArgmax([l, l])).phonemes;
      expect(out, ['L']);
    });

    test('blanks and specials are dropped', () {
      final out =
          greedyDecode(lpFromArgmax([l, blank, special, l, k, k])).phonemes;
      expect(out, ['L', 'L', 'K']);
    });

    test('all-blank frames decode to empty with blankFraction 1.0', () {
      final d = greedyDecode(lpFromArgmax([blank, blank, blank]));
      expect(d.phonemes, isEmpty);
      expect(d.blankFraction, 1.0);
    });

    test('blankFraction counts only blank-argmax frames', () {
      final d = greedyDecode(lpFromArgmax([l, blank, k, blank]));
      expect(d.blankFraction, closeTo(0.5, 1e-9));
    });
  });
}
