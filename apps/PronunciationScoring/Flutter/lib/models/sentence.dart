/// A bundled practice sentence with its precomputed ARPABET target.
///
/// Phonemes are computed OFFLINE (CMUdict, stress digits stripped, first
/// pronunciation) by tools/gen_sentences.py — there is no runtime G2P. The
/// per-word phone [spans] partition [phonemeIds] exactly: word i owns the
/// phones in [spans[i].start, spans[i].end).
class PracticeSentence {
  const PracticeSentence({
    required this.text,
    required this.words,
    required this.phonemeIds,
    required this.spans,
    required this.estSeconds,
  });

  /// Display text of the sentence.
  final String text;

  /// Word tokens, in order (aligned 1:1 with [spans]).
  final List<String> words;

  /// Flat ARPABET id sequence (ids 0..38) for the whole sentence.
  final List<int> phonemeIds;

  /// Per-word [start, end) slice into [phonemeIds].
  final List<PhoneSpan> spans;

  /// Offline read-time estimate (seconds); curated into the 3.5–5.0 s band.
  final double estSeconds;

  factory PracticeSentence.fromJson(Map<String, dynamic> json) {
    return PracticeSentence(
      text: json['text'] as String,
      words: (json['words'] as List).cast<String>(),
      phonemeIds: (json['phoneme_ids'] as List).map((e) => e as int).toList(),
      spans: (json['spans'] as List)
          .map((e) => PhoneSpan(
                (e as List)[0] as int,
                e[1] as int,
              ))
          .toList(),
      estSeconds: (json['est_seconds'] as num).toDouble(),
    );
  }
}

/// A half-open [start, end) range of phone indices owned by one word.
class PhoneSpan {
  const PhoneSpan(this.start, this.end);
  final int start;
  final int end;
  int get length => end - start;
}
