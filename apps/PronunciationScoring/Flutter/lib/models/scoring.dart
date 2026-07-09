/// Scoring result value types produced by the pure-Dart scoring head.
library;

/// GOP result for one target phoneme.
class PhonemeScore {
  const PhonemeScore({
    required this.phoneme,
    required this.id,
    required this.gop,
    required this.score,
    required this.alignedFrames,
  });

  /// ARPABET label (e.g. "AA").
  final String phoneme;

  /// Class id (0..38).
  final int id;

  /// Raw goodness-of-pronunciation: mean posterior over aligned frames (0..1).
  /// 0.0 when the phoneme received no aligned frames.
  final double gop;

  /// Fill-aware calibrated 0..100 score.
  final double score;

  /// Number of CTC frames the aligner assigned to this phoneme.
  final int alignedFrames;
}

/// Aggregated score for one word.
class WordScore {
  const WordScore({
    required this.word,
    required this.phonemes,
    required this.score,
    required this.weakestPhonemeIndex,
  });

  final String word;
  final List<PhonemeScore> phonemes;

  /// Fill-aware calibrated 0..100 (mean of this word's phoneme scores).
  final double score;

  /// Index into [phonemes] of the lowest-scoring phone (the "fix this sound"
  /// highlight), or -1 if the word has no phonemes.
  final int weakestPhonemeIndex;

  PhonemeScore? get weakestPhoneme =>
      weakestPhonemeIndex < 0 ? null : phonemes[weakestPhonemeIndex];
}

/// Full result of scoring one recording against one target sentence.
class PronunciationResult {
  const PronunciationResult({
    required this.words,
    required this.overallScore,
    required this.blankFraction,
    required this.greedyPhonemes,
    required this.inferenceMs,
    required this.scoringMs,
    required this.sampleRateInfo,
  });

  final List<WordScore> words;

  /// Overall fill-aware calibrated 0..100.
  final double overallScore;

  /// Fraction of CTC frames whose argmax is the blank class (window-fill proxy).
  final double blankFraction;

  /// Greedy CTC decode ("what we heard") — decoration only; never used to score.
  final List<String> greedyPhonemes;

  /// Wall-clock model inference latency (ms).
  final int inferenceMs;

  /// Wall-clock scoring-head latency (ms).
  final int scoringMs;

  /// Human-readable capture/decimation note for the HUD.
  final String sampleRateInfo;
}
