import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:livedocredact/services/ctc_decoder.dart';

/// Builds a flattened [1,40,438] probability tensor from a per-step class
/// list ([classPerStep] shorter than 40 is padded with blanks), assigning
/// probability [p] to the chosen class and ~0 elsewhere.
Float32List makeProbs(List<int> classPerStep,
    {double p = 0.9, double floor = 0.0001}) {
  final probs = Float32List(kCtcSteps * kCtcClasses);
  for (var i = 0; i < probs.length; i++) {
    probs[i] = floor;
  }
  for (var t = 0; t < kCtcSteps; t++) {
    final cls = t < classPerStep.length ? classPerStep[t] : kCtcBlankIndex;
    probs[t * kCtcClasses + cls] = p;
  }
  return probs;
}

void main() {
  // The REAL charset the app ships (assets/en_dict.txt) — the decoder must be
  // built from it in exactly the SPEC order: [blank] + dict(1..436) + ' '.
  final dictRaw = File('assets/en_dict.txt').readAsStringSync();
  final decoder = CtcDecoder.fromDictString(dictRaw);

  int cls(String ch) {
    final idx = decoder.labels.indexOf(ch);
    expect(idx, greaterThan(0), reason: 'char "$ch" must exist in the dict');
    return idx;
  }

  group('label list construction (SPEC-binding order)', () {
    test('exactly 438 classes: [blank] + 436 dict chars + space', () {
      expect(decoder.labels.length, kCtcClasses);
      expect(decoder.labels[0], '', reason: 'blank is class 0');
      expect(decoder.labels[kCtcClasses - 1], ' ',
          reason: 'space is the LAST class (437)');
    });

    test('dict line i (1-based) is class i', () {
      final lines = dictRaw.trimRight().split('\n');
      expect(lines.length, 436);
      expect(decoder.labels[1], lines[0], reason: 'first dict line -> class 1');
      expect(decoder.labels[436], lines[435],
          reason: 'last dict line -> class 436');
      // '<' (the MRZ filler) is dict line 80 -> class 80.
      expect(decoder.labels[80], '<');
    });

    test('wrong dict size is rejected loudly', () {
      expect(() => CtcDecoder(List.filled(10, 'x')), throwsArgumentError);
    });
  });

  group('greedy decode semantics', () {
    test('collapse repeats THEN drop blank yields the encoded string', () {
      final a = cls('A'), b = cls('B'), one = cls('1');
      // "A A B _ B 1 1 _" (underscore = blank): repeats collapse, blanks drop.
      final probs = makeProbs([a, a, b, 0, b, one, one, 0]);
      expect(decoder.decode(probs).text, 'ABB1');
    });

    test('a genuine double letter survives when separated by a blank', () {
      final a = cls('A');
      expect(decoder.decode(makeProbs([a, 0, a])).text, 'AA');
      expect(decoder.decode(makeProbs([a, a])).text, 'A',
          reason: 'consecutive identical classes collapse');
    });

    test('space (class 437) decodes to a real space', () {
      final a = cls('A'), b = cls('B');
      final probs = makeProbs([a, kCtcClasses - 1, b]);
      expect(decoder.decode(probs).text, 'A B');
    });

    test('all-blank tensor decodes to empty with zero confidence', () {
      final result = decoder.decode(makeProbs([]));
      expect(result.text, isEmpty);
      expect(result.confidence, 0.0);
    });
  });

  group('output tensor layout [1,40,438]', () {
    test('argmax runs over the LAST axis (classes) per step', () {
      final a = cls('A'), b = cls('B'), c = cls('C');
      final probs = makeProbs([a, 0, b, 0, c]);
      expect(decoder.decode(probs).text, 'ABC');

      // A step-major/class-major stride confusion reads a transposed tensor.
      // The same data laid out transposed must NOT decode to the same string.
      final transposed = Float32List(kCtcSteps * kCtcClasses);
      for (var t = 0; t < kCtcSteps; t++) {
        for (var cc = 0; cc < kCtcClasses; cc++) {
          // transposed[c * steps + t] = probs[t * classes + c] for the region
          // that fits; the rest stays 0 (an invalid layout, which is the point).
          final int dst = cc * kCtcSteps + t;
          if (dst < transposed.length) {
            transposed[dst] = probs[t * kCtcClasses + cc];
          }
        }
      }
      expect(decoder.decode(transposed).text, isNot('ABC'),
          reason: 'a wrong-axis read must not silently produce the right text');
    });
  });

  group('activation semantics', () {
    test('confidence is the raw mean max-probability — no extra activation',
        () {
      final a = cls('A'), b = cls('B');
      final probs = makeProbs([a], p: 0.7);
      // One emitted char with raw prob 0.7: confidence must be exactly 0.7
      // (a spurious sigmoid would give ~0.668, a softmax something else).
      expect(decoder.decode(probs).confidence, closeTo(0.7, 1e-6));

      // Two chars at 0.6 / 0.8 -> mean 0.7 over EMITTED steps only.
      final probs2 = makeProbs([a, 0, b], p: 0.6);
      probs2[2 * kCtcClasses + b] = 0.8;
      expect(decoder.decode(probs2).confidence, closeTo(0.7, 1e-6));
    });
  });
}
