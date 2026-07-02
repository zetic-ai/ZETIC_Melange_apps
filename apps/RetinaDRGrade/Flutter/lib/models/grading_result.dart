/// The result of one on-device diabetic-retinopathy SEVERITY grading.
///
/// This is a 5-GRADE grader (unlike the sibling app RetinaDRScreen, which is the
/// binary referable screener): [grade] is an integer 0..4 and [perGradeProbs] is
/// the full 5-way softmax distribution. Referable = grade >= 2.
class GradingResult {
  const GradingResult({
    required this.grade,
    required this.perGradeProbs,
    required this.referable,
    required this.logits,
  });

  /// The predicted DR severity grade, an integer 0..4. This is the argmax of
  /// [perGradeProbs] used directly as the grade (id2label is the identity map,
  /// no remap).
  final int grade;

  /// The full 5-way softmax distribution (length 5), one probability per grade
  /// 0..4. Sums to ~1. Feeds the 5-bar confidence widget.
  final List<double> perGradeProbs;

  /// True if the eye should be referred: grade >= 2 (Moderate / Severe /
  /// Proliferative). No DR (0) and Mild (1) are NOT referable.
  final bool referable;

  /// The five raw model logits (length 5), surfaced for the on-screen HUD.
  final List<double> logits;

  /// The five canonical grade labels, index == grade (identity id2label).
  static const List<String> gradeLabels = [
    'No DR',
    'Mild',
    'Moderate',
    'Severe',
    'Proliferative',
  ];

  /// Grade at or above which an eye is referable.
  static const int referableGrade = 2;

  /// Human label for the predicted grade, e.g. "Severe".
  String get gradeLabel => gradeLabels[grade];

  /// Primary readout, e.g. "Grade 3 — Severe".
  String get gradeHeadline => 'Grade $grade — $gradeLabel';

  /// Top-1 confidence = the probability of the predicted grade.
  double get topConfidence => perGradeProbs[grade];

  /// Referable banner text.
  String get referableLabel => referable ? 'REFERABLE' : 'NOT REFERABLE';

  @override
  String toString() =>
      'GradingResult(grade: $grade ($gradeLabel), '
      'referable: $referable, '
      'probs: [${perGradeProbs.map((p) => p.toStringAsFixed(3)).join(', ')}], '
      'logits: [${logits.map((l) => l.toStringAsFixed(2)).join(', ')}])';
}
