import 'dart:math' as math;

import '../models/phonemes.dart';
import '../models/scoring.dart';
import '../models/sentence.dart';
import 'ctc_aligner.dart';
import 'postprocessor.dart';

/// Pure-Dart goodness-of-pronunciation (GOP) scoring head.
///
/// Per-phoneme GOP = mean over the phoneme's aligned frames of exp(logprob) of
/// the TARGET id (posterior of the phone we expected). Phones with no aligned
/// frames score 0. Word score = mean of its phones' calibrated scores, and the
/// lowest phone is surfaced as the "fix this sound" highlight. Overall = mean
/// of word scores.
///
/// Fill-aware calibration: GOP is naturally depressed when the speaker fills
/// less of the 5.11 s window (measured on the reference clips —
/// see validation/). The blank-frame fraction is a usable window-fill proxy, so
/// the raw GOP is normalized against the GOP a good speaker is expected to
/// reach at that fill before mapping to 0..100.
class GopScorer {
  const GopScorer({this.aligner = const CtcAligner()});

  final CtcAligner aligner;

  // --- Fill-aware calibration anchors (measured, see validation/) ---
  // ls1  : blank_frac 0.19, good-speaker GOP ~0.75 (94% window fill)
  // ref2 : blank_frac 0.48, good-speaker GOP ~0.19 (39% window fill)
  static const double _fullFillBlank = 0.20;
  static const double _lowFillBlank = 0.48;
  static const double _expGoodAtFull = 0.75;
  static const double _expGoodAtLow = 0.20;

  /// Expected good-speaker GOP at a given blank-frame fraction (window fill).
  /// Higher blank fraction (less speech) -> lower reachable GOP.
  static double expectedGoodGop(double blankFraction) {
    if (blankFraction <= _fullFillBlank) return _expGoodAtFull;
    if (blankFraction >= _lowFillBlank) return _expGoodAtLow;
    final tt = (blankFraction - _fullFillBlank) / (_lowFillBlank - _fullFillBlank);
    return _expGoodAtFull + tt * (_expGoodAtLow - _expGoodAtFull);
  }

  /// Map a raw GOP (0..1) to a fill-aware 0..100 score.
  static double calibrate(double gop, double blankFraction) {
    final exp = expectedGoodGop(blankFraction);
    final norm = exp <= 0 ? 0.0 : gop / exp;
    return norm.clamp(0.0, 1.0) * 100.0;
  }

  /// Score one recording's [lp] against [sentence]. [blankFraction] is the
  /// window-fill proxy from the greedy decode.
  List<WordScore> scoreWords(
    LogProbView lp,
    PracticeSentence sentence,
    double blankFraction,
  ) {
    final frames = aligner.align(lp, sentence.phonemeIds);

    // Per-phoneme GOP over the whole sentence.
    final phoneScores = List<PhonemeScore>.generate(
      sentence.phonemeIds.length,
      (i) {
        final id = sentence.phonemeIds[i];
        final fr = frames[i];
        double gop = 0.0;
        if (fr.isNotEmpty) {
          var sum = 0.0;
          for (final f in fr) {
            sum += math.exp(lp.at(f, id));
          }
          gop = sum / fr.length;
        }
        return PhonemeScore(
          phoneme: Phonemes.arpabet[id],
          id: id,
          gop: gop,
          score: calibrate(gop, blankFraction),
          alignedFrames: fr.length,
        );
      },
      growable: false,
    );

    // Aggregate into words using the sentence's phone spans.
    return List<WordScore>.generate(sentence.words.length, (w) {
      final span = sentence.spans[w];
      final phones = phoneScores.sublist(span.start, span.end);
      var sum = 0.0;
      var weakest = -1;
      var weakestScore = double.infinity;
      for (var i = 0; i < phones.length; i++) {
        sum += phones[i].score;
        if (phones[i].score < weakestScore) {
          weakestScore = phones[i].score;
          weakest = i;
        }
      }
      final wordScore = phones.isEmpty ? 0.0 : sum / phones.length;
      return WordScore(
        word: sentence.words[w],
        phonemes: phones,
        score: wordScore,
        weakestPhonemeIndex: weakest,
      );
    }, growable: false);
  }

  /// Overall sentence score = mean of word scores (0..100).
  double overall(List<WordScore> words) {
    if (words.isEmpty) return 0.0;
    var sum = 0.0;
    for (final w in words) {
      sum += w.score;
    }
    return sum / words.length;
  }
}
