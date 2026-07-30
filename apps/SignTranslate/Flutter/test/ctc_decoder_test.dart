import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:signtranslate/config.dart';
import 'package:signtranslate/services/ctc_decoder.dart';

/// Builds a [kRecTimeSteps]×[kRecNumClasses] probability tensor from a list
/// of (classIndex, probability) per time step; all other classes get a small
/// uniform floor so the argmax is unambiguous.
Float32List tensorFromSteps(List<(int, double)> steps) {
  final out = Float32List(kRecTimeSteps * kRecNumClasses);
  for (var t = 0; t < kRecTimeSteps; t++) {
    final (idx, p) = t < steps.length ? steps[t] : (0, 0.9); // pad = blank
    for (var c = 0; c < kRecNumClasses; c++) {
      out[t * kRecNumClasses + c] = c == idx ? p : 0.0001;
    }
  }
  return out;
}

void main() {
  // The REAL shipped charset — the exact asset the app loads.
  final charsetText =
      File('assets/charset/latin_charset.txt').readAsStringSync();
  final decoder = CtcDecoder.fromCharsetText(charsetText);
  final lines = charsetText.split('\n')..removeWhere((l) => l.isEmpty);

  int classOf(String ch) {
    final i = lines.indexOf(ch);
    expect(i, greaterThanOrEqualTo(0), reason: 'charset must contain "$ch"');
    return i + 1; // charset line i -> class index i+1 (blank occupies 0)
  }

  group('838-class charset map (the #1 silent-wrong trap)', () {
    test('map has EXACTLY 838 classes — not 438, not 837, not 839', () {
      expect(decoder.classes.length, kRecNumClasses);
      expect(kRecNumClasses, 838);
    });

    test('blank occupies index 0', () {
      expect(decoder.classes[0], '');
    });

    test('index i (1..836) maps to charset line i-1 — off-by-one pinned', () {
      expect(decoder.classes[1], lines[0]); // first line is '0'
      expect(decoder.classes[1], '0');
      expect(decoder.classes[836], lines[835]); // last charset line
      expect(decoder.classes[11], 'A'); // line 11 of the file
      expect(decoder.classes[37], 'a');
    });

    test('index 837 is space', () {
      expect(decoder.classes[837], ' ');
    });

    test('a wrong-size charset throws loudly instead of shifting chars', () {
      expect(
        () => CtcDecoder.fromCharsetLines(List.filled(437, 'x')),
        throwsStateError,
      );
      expect(
        () => CtcDecoder.fromCharsetLines(List.filled(837, 'x')),
        throwsStateError,
      );
    });
  });

  group('greedy CTC decode semantics', () {
    test('decodes a known word with a space (837)', () {
      final h = classOf('H'), i = classOf('I');
      final result = decoder.decode(tensorFromSteps([
        (0, 0.9), // blank
        (h, 0.8),
        (i, 0.7),
        (837, 0.6), // space
        (h, 0.9),
        (i, 0.5),
      ]));
      expect(result.text, 'HI HI');
    });

    test('consecutive duplicates merge; blank separates repeats', () {
      final h = classOf('H'), i = classOf('I');
      // blank,H,H,blank,I -> "HI"
      expect(
        decoder
            .decode(tensorFromSteps([(0, .9), (h, .8), (h, .8), (0, .9), (i, .7)]))
            .text,
        'HI',
      );
      // H,blank,H -> "HH" (NOT merged across the blank)
      expect(
        decoder.decode(tensorFromSteps([(h, .8), (0, .9), (h, .8)])).text,
        'HH',
      );
    });

    test('blanks are dropped, all-blank decodes to empty', () {
      final result = decoder.decode(tensorFromSteps([]));
      expect(result.text, isEmpty);
      expect(result.confidence, 0.0);
    });

    test('confidence = mean max-prob over EMITTED steps (hand-computed)', () {
      final h = classOf('H'), i = classOf('I');
      // Emitted: H@0.8 (t1) and I@0.6 (t3). The duplicate H@0.4 (t2) and the
      // blanks contribute NOTHING.
      final result = decoder.decode(tensorFromSteps([
        (0, 0.9),
        (h, 0.8),
        (h, 0.4),
        (i, 0.6),
      ]));
      expect(result.text, 'HI');
      expect(result.confidence, closeTo((0.8 + 0.6) / 2, 1e-6));
    });

    test('accented Latin characters survive the round trip', () {
      const word = 'Éé';
      final e1 = classOf('É'), e2 = classOf('é');
      final result = decoder.decode(tensorFromSteps([(e1, .9), (e2, .9)]));
      expect(result.text, word);
    });
  });
}
