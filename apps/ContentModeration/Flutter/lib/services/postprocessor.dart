import 'dart:math' as math;

import '../models/moderation_result.dart';

/// Pure-Dart post-processing for the binary NSFW/SFW content-safety gate.
///
/// The ONNX graph outputs RAW LOGITS `float32[1,2]` with the semantic layout
/// `[0] = NSFW logit, [1] = SFW logit` — index 0 is NSFW, the REVERSE of the
/// usual "0 = safe" convention. A swapped index inverts every decision. Softmax
/// is NOT baked into the graph and must be applied exactly ONCE here.
/// See SPEC_STUB.md "Post-processing pipeline" / "Validation focus".
class Postprocessor {
  const Postprocessor({
    this.reviewThreshold = defaultReviewThreshold,
    this.blockThreshold = defaultBlockThreshold,
  });

  /// P(NSFW) at/above which an image leaves KEEP and enters REVIEW/BLUR.
  static const double defaultReviewThreshold = 0.30;

  /// P(NSFW) at/above which an image enters BLOCK.
  static const double defaultBlockThreshold = 0.70;

  /// Output index of the NSFW logit (idx0 = NSFW, idx1 = SFW).
  static const int nsfwIndex = 0;

  /// Lower band boundary (inclusive for REVIEW): pNsfw >= this leaves KEEP.
  final double reviewThreshold;

  /// Upper band boundary (inclusive for BLOCK): pNsfw >= this is BLOCK.
  final double blockThreshold;

  /// Numerically-stable softmax over the 2 logits.
  ///
  /// Subtracting the max before exponentiating avoids overflow without changing
  /// the result. Returns `[P(NSFW), P(SFW)]`, summing to 1.
  static List<double> softmax(List<double> logits) {
    if (logits.length != 2) {
      throw ArgumentError(
        'Expected exactly 2 logits [NSFW, SFW], got ${logits.length}.',
      );
    }
    final maxLogit = math.max(logits[0], logits[1]);
    final e0 = math.exp(logits[0] - maxLogit);
    final e1 = math.exp(logits[1] - maxLogit);
    final sum = e0 + e1;
    return [e0 / sum, e1 / sum];
  }

  /// Map a P(NSFW) value to its decision band under the active thresholds.
  Decision bandFor(double pNsfw) {
    if (pNsfw >= blockThreshold) return Decision.block;
    if (pNsfw >= reviewThreshold) return Decision.review;
    return Decision.keep;
  }

  /// Turn the raw `[NSFW, SFW]` logits into a [ModerationResult].
  ModerationResult classify(List<double> logits) {
    final probs = softmax(logits);
    final pNsfw = probs[nsfwIndex];
    final pSfw = probs[1 - nsfwIndex];
    return ModerationResult(
      pNsfw: pNsfw,
      pSfw: pSfw,
      decision: bandFor(pNsfw),
      logits: List<double>.unmodifiable(logits),
    );
  }
}
