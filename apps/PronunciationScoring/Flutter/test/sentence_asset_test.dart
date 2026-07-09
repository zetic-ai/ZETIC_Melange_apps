import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:sayright/models/phonemes.dart';
import 'package:sayright/services/sentence_repository.dart';

// Must match SEC_PER_PHONEME in tools/gen_sentences.py.
const double kSecondsPerPhoneme = 0.105;

void main() {
  final sentences =
      SentenceRepository.parse(File('assets/sentences.json').readAsStringSync());

  test('the asset bundles a handful of sentences', () {
    expect(sentences.length, greaterThanOrEqualTo(6));
  });

  for (final s in sentences) {
    group('sentence: "${s.text}"', () {
      test('every phoneme id is a real phoneme (0..38)', () {
        expect(s.phonemeIds, isNotEmpty);
        for (final id in s.phonemeIds) {
          expect(Phonemes.isPhoneme(id), isTrue, reason: 'id $id out of 0..38');
        }
      });

      test('word spans exactly partition the phoneme sequence', () {
        expect(s.spans.length, s.words.length);
        var cursor = 0;
        for (final span in s.spans) {
          expect(span.start, cursor, reason: 'no gap/overlap between words');
          expect(span.end, greaterThan(span.start), reason: 'non-empty word');
          cursor = span.end;
        }
        expect(cursor, s.phonemeIds.length,
            reason: 'spans cover all phonemes');
      });

      test('read-time estimate is in the 3.5–5.0 s window-fill band', () {
        expect(s.estSeconds, inInclusiveRange(3.5, 5.0));
        // Consistent with the shared phoneme->seconds model.
        expect(s.estSeconds,
            closeTo(s.phonemeIds.length * kSecondsPerPhoneme, 1e-3));
      });
    });
  }
}
