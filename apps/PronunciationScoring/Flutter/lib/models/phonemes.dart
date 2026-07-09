/// ARPABET phoneme label table for citrinet256_phoneme.onnx.
///
/// AUTHORITATIVE contract (see app-root labels.txt): ids 0..38 are ARPABET
/// phonemes (AA=0 ... ZH=38), ids 39..43 are unused tokenizer specials that
/// never fire on real speech, and id 44 is the CTC blank.
///
/// TRAP: NeMo's own detokenizer DISPLAYS id 0 as "[PAD]" — that map is a
/// repackaging artifact and is WRONG. Empirically id 0 is AA. Trust this table.
class Phonemes {
  Phonemes._();

  /// ids 0..38, in vocab order. Index == class id.
  static const List<String> arpabet = <String>[
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER',
    'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH',
  ];

  /// Number of ARPABET phoneme classes (ids 0..38).
  static const int phonemeCount = 39;

  /// First unused tokenizer special id ([UNK]). ids 39..43 never fire.
  static const int firstSpecial = 39;

  /// CTC blank id.
  static const int blank = 44;

  /// Total output classes per CTC frame.
  static const int classCount = 45;

  /// Number of CTC frames the model emits for one 5.11 s window.
  static const int frameCount = 64;

  /// Map an ARPABET label to its class id, or -1 if unknown.
  static int idOf(String phoneme) => arpabet.indexOf(phoneme);

  /// True for a scorable phoneme id (0..38); false for specials/blank.
  static bool isPhoneme(int id) => id >= 0 && id < phonemeCount;
}
