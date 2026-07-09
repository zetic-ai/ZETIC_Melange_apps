import 'package:flutter_test/flutter_test.dart';
import 'package:sayright/models/phonemes.dart';

void main() {
  group('ARPABET label table (labels.txt is authoritative)', () {
    test('id 0 is AA, NOT [PAD] (the NeMo display-map trap)', () {
      expect(Phonemes.arpabet[0], 'AA');
      expect(Phonemes.arpabet[0], isNot('[PAD]'));
    });

    test('id 38 is ZH (last phoneme)', () {
      expect(Phonemes.arpabet[38], 'ZH');
    });

    test('39 phoneme classes, blank at 44, 45 classes total', () {
      expect(Phonemes.phonemeCount, 39);
      expect(Phonemes.arpabet.length, 39);
      expect(Phonemes.blank, 44);
      expect(Phonemes.classCount, 45);
    });

    test('specials 39..43 and blank 44 are excluded from phonemes', () {
      for (var id = 39; id <= 44; id++) {
        expect(Phonemes.isPhoneme(id), isFalse, reason: 'id $id must not score');
      }
      for (var id = 0; id < 39; id++) {
        expect(Phonemes.isPhoneme(id), isTrue);
      }
    });

    test('a few known ids round-trip', () {
      expect(Phonemes.idOf('AO'), 3);
      expect(Phonemes.idOf('K'), 19);
      expect(Phonemes.idOf('SH'), 29);
    });
  });
}
