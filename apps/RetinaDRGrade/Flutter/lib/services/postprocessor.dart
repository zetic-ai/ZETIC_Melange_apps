import 'dart:math' as math;

import '../models/grading_result.dart';

/// Pure-Dart post-processing for the 5-class ViT DR severity grader.
///
/// The ONNX graph outputs RAW LOGITS `float32[1,5]` with semantic layout
/// `logits[i]` = the unnormalized score for DR grade `i` (0..4). Softmax is NOT
/// baked into the graph and must be applied exactly ONCE here. The argmax index
/// IS the canonical grade directly (id2label is the identity map {0..4}) — no
/// remap, no inversion, no reorder. See SPEC.md "Post-processing pipeline".
class Postprocessor {
  const Postprocessor();

  /// Number of grades / logits (0 No DR .. 4 Proliferative).
  static const int numGrades = 5;

  /// Numerically-stable softmax over the 5 logits.
  ///
  /// Subtracting the max before exponentiating avoids overflow and does not
  /// change the result. Returns a 5-vector of per-grade probabilities summing
  /// to 1. Applied exactly ONCE (the graph does not softmax).
  static List<double> softmax(List<double> logits) {
    if (logits.length != numGrades) {
      throw ArgumentError(
        'Expected exactly $numGrades logits (one per grade 0..4), '
        'got ${logits.length}.',
      );
    }
    var maxLogit = logits[0];
    for (final l in logits) {
      if (l > maxLogit) maxLogit = l;
    }
    final exps = List<double>.filled(numGrades, 0.0);
    var sum = 0.0;
    for (var i = 0; i < numGrades; i++) {
      final e = math.exp(logits[i] - maxLogit);
      exps[i] = e;
      sum += e;
    }
    for (var i = 0; i < numGrades; i++) {
      exps[i] /= sum;
    }
    return exps;
  }

  /// Index of the largest element (argmax). Ties resolve to the lowest index.
  static int argmax(List<double> values) {
    var best = 0;
    for (var i = 1; i < values.length; i++) {
      if (values[i] > values[best]) best = i;
    }
    return best;
  }

  /// Turn the raw 5 logits into a [GradingResult].
  ///
  /// softmax ONCE -> argmax = grade (identity id2label) -> referable = grade >= 2.
  GradingResult classify(List<double> logits) {
    final probs = softmax(logits);
    final grade = argmax(probs);
    final referable = grade >= GradingResult.referableGrade;
    return GradingResult(
      grade: grade,
      perGradeProbs: List<double>.unmodifiable(probs),
      referable: referable,
      logits: List<double>.unmodifiable(logits),
    );
  }
}
