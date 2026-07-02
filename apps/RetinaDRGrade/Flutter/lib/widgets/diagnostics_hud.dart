import 'package:flutter/material.dart';

import '../models/grading_result.dart';
import '../services/preprocessor.dart';
import '../theme.dart';

/// On-screen diagnostics. In a release device build Dart `print`/`debugPrint`
/// does NOT reliably reach the native console, so per-inference latency, the five
/// raw logits, the softmax vector, and the input tensor shape are surfaced here
/// on the UI/HUD (CLAUDE.md §5).
class DiagnosticsHud extends StatelessWidget {
  const DiagnosticsHud({
    super.key,
    required this.result,
    required this.preprocessMs,
    required this.inferenceMs,
  });

  final GradingResult result;
  final double preprocessMs;
  final double inferenceMs;

  @override
  Widget build(BuildContext context) {
    final shape = Preprocessor.tensorShape.join('×');
    final logits = result.logits.map((l) => l.toStringAsFixed(3)).join(', ');
    final softmax =
        result.perGradeProbs.map((p) => p.toStringAsFixed(3)).join(', ');
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: GradeVueTheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: GradeVueTheme.surfaceAlt),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'DIAGNOSTICS',
            style: TextStyle(
              color: GradeVueTheme.onSurfaceMuted,
              fontSize: 11,
              fontWeight: FontWeight.w700,
              letterSpacing: 1.2,
            ),
          ),
          const SizedBox(height: 8),
          _row('Inference', '${inferenceMs.toStringAsFixed(1)} ms'),
          _row('Preprocess (Dart)', '${preprocessMs.toStringAsFixed(1)} ms'),
          _row('Input tensor', 'float32[$shape] NCHW RGB'),
          _row('Predicted grade', '${result.grade} (${result.gradeLabel})'),
          _row('Raw logits [0..4]', '[$logits]'),
          _row('Softmax [0..4]', '[$softmax]'),
        ],
      ),
    );
  }

  Widget _row(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            flex: 5,
            child: Text(
              label,
              style: const TextStyle(
                color: GradeVueTheme.onSurfaceMuted,
                fontSize: 12,
              ),
            ),
          ),
          Expanded(
            flex: 6,
            child: Text(
              value,
              textAlign: TextAlign.right,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 12,
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
