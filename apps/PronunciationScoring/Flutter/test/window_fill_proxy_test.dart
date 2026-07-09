import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sayright/models/phonemes.dart';
import 'package:sayright/services/gop_scorer.dart';
import 'package:sayright/services/postprocessor.dart';

LogProbView lpFromArgmax(List<int> ids) {
  const c = Phonemes.classCount;
  final data = Float32List(ids.length * c)..fillRange(0, ids.length * c, -5.0);
  for (var f = 0; f < ids.length; f++) {
    data[f * c + ids[f]] = 5.0;
  }
  return LogProbView(data, frames: ids.length, classes: c);
}

void main() {
  const blank = Phonemes.blank;

  group('blank fraction as window-fill proxy', () {
    test('hand-built frames: 3 of 4 blank -> 0.75', () {
      final d = greedyDecode(lpFromArgmax([20, blank, blank, blank]));
      expect(d.blankFraction, closeTo(0.75, 1e-9));
    });

    test('no blanks -> 0.0; all blanks -> 1.0', () {
      expect(greedyDecode(lpFromArgmax([20, 19, 22])).blankFraction, 0.0);
      expect(greedyDecode(lpFromArgmax([blank, blank])).blankFraction, 1.0);
    });
  });

  group('fill-aware expectedGoodGop boundaries', () {
    test('anchors: full fill 0.75, low fill 0.20', () {
      expect(GopScorer.expectedGoodGop(0.20), closeTo(0.75, 1e-9));
      expect(GopScorer.expectedGoodGop(0.48), closeTo(0.20, 1e-9));
    });

    test('clamps outside the calibrated band', () {
      expect(GopScorer.expectedGoodGop(0.05), 0.75); // below -> full-fill anchor
      expect(GopScorer.expectedGoodGop(0.90), 0.20); // above -> low-fill anchor
    });

    test('monotonically decreasing across the band', () {
      final mid = GopScorer.expectedGoodGop(0.34);
      expect(mid, lessThan(0.75));
      expect(mid, greaterThan(0.20));
    });
  });

  group('calibrate() maps GOP -> 0..100 with fill compensation', () {
    test('a good full-fill GOP scores ~100; a mismatched one scores low', () {
      expect(GopScorer.calibrate(0.75, 0.19), closeTo(100.0, 1e-6));
      expect(GopScorer.calibrate(0.14, 0.19), lessThan(25.0));
    });

    test('output is clamped to [0, 100]', () {
      expect(GopScorer.calibrate(2.0, 0.19), 100.0);
      expect(GopScorer.calibrate(0.0, 0.19), 0.0);
    });

    test('the SAME raw GOP scores higher at low fill (compensation)', () {
      final atFull = GopScorer.calibrate(0.30, 0.20);
      final atLow = GopScorer.calibrate(0.30, 0.45);
      expect(atLow, greaterThan(atFull));
    });
  });
}
