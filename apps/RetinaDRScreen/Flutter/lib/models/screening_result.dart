/// The result of one on-device diabetic-retinopathy screening.
///
/// This is a BINARY screener: [referable] is the whole verdict. There is no
/// 0–4 severity grade (that is the sibling app RetinaDRGrade).
class ScreeningResult {
  const ScreeningResult({
    required this.referable,
    required this.pReferable,
    required this.confidence,
    required this.logits,
  });

  /// True if the eye should be referred (DR grade >= 2). Equivalent to
  /// `pReferable >= threshold` (default threshold 0.5).
  final bool referable;

  /// P(referable) = softmax(logits)[1], in [0, 1].
  final double pReferable;

  /// Confidence of the SHOWN verdict = max(P0, P1), in [0.5, 1].
  final double confidence;

  /// The two raw model logits `[Nrdr, Rdr]`, surfaced for the on-screen HUD.
  final List<double> logits;

  /// P(not-referable) = 1 - P(referable).
  double get pNotReferable => 1.0 - pReferable;

  /// Primary banner text.
  String get verdictLabel => referable ? 'REFERABLE' : 'NOT REFERABLE';

  @override
  String toString() =>
      'ScreeningResult(referable: $referable, '
      'pReferable: ${pReferable.toStringAsFixed(4)}, '
      'confidence: ${confidence.toStringAsFixed(4)}, '
      'logits: [${logits.map((l) => l.toStringAsFixed(2)).join(', ')}])';
}
