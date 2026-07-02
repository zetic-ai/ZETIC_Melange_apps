import 'dart:math' as math;

import '../models/screening_result.dart';

/// Pure-Dart post-processing for the binary DR screener.
///
/// The ONNX graph outputs RAW LOGITS `float32[1,2]` with semantic layout
/// `[0] = Nrdr logit, [1] = Rdr logit`. Softmax is NOT baked into the graph and
/// must be applied exactly ONCE here. See SPEC.md "Post-processing pipeline".
class Postprocessor {
  const Postprocessor({this.threshold = defaultThreshold});

  /// Shipped default decision threshold. 0.5 is argmax-equivalent on 2 logits.
  /// This build ships a FIXED 0.5 (no sensitivity slider) so the booth demo is
  /// unambiguous.
  static const double defaultThreshold = 0.5;

  /// Label index for the "referable" (Rdr) class.
  static const int referableIndex = 1;

  final double threshold;

  /// Numerically-stable softmax over the 2 logits.
  ///
  /// Subtracting the max before exponentiating avoids overflow and does not
  /// change the result. Returns `[P(not-referable), P(referable)]`, summing to 1.
  static List<double> softmax(List<double> logits) {
    if (logits.length != 2) {
      throw ArgumentError(
        'Expected exactly 2 logits [Nrdr, Rdr], got ${logits.length}.',
      );
    }
    final maxLogit = math.max(logits[0], logits[1]);
    final e0 = math.exp(logits[0] - maxLogit);
    final e1 = math.exp(logits[1] - maxLogit);
    final sum = e0 + e1;
    return [e0 / sum, e1 / sum];
  }

  /// Turn the raw `[Nrdr, Rdr]` logits into a [ScreeningResult].
  ///
  /// P(referable) = softmax[1]; verdict is REFERABLE iff P(referable) >= threshold.
  /// Confidence of the shown verdict = max(P0, P1).
  ScreeningResult classify(List<double> logits) {
    final probs = softmax(logits);
    final pReferable = probs[referableIndex];
    final referable = pReferable >= threshold;
    final confidence = math.max(probs[0], probs[1]);
    return ScreeningResult(
      referable: referable,
      pReferable: pReferable,
      confidence: confidence,
      logits: List<double>.unmodifiable(logits),
    );
  }
}
