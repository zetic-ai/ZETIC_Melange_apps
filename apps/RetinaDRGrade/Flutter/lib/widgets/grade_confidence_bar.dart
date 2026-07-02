import 'package:flutter/material.dart';

import '../models/grading_result.dart';
import '../theme.dart';

/// The full 5-way severity distribution: one horizontal bar per grade 0..4, each
/// colored by its severity, with the predicted (argmax) grade highlighted. Shows
/// the whole distribution, not just the top-1. Grades >= 2 (referable) are marked.
class GradeConfidenceBar extends StatelessWidget {
  const GradeConfidenceBar({super.key, required this.result});

  final GradingResult result;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'PER-GRADE CONFIDENCE',
          style: TextStyle(
            color: GradeVueTheme.onSurfaceMuted,
            fontSize: 11,
            fontWeight: FontWeight.w700,
            letterSpacing: 1.2,
          ),
        ),
        const SizedBox(height: 10),
        for (var g = 0; g < GradingResult.gradeLabels.length; g++)
          _GradeRow(
            grade: g,
            prob: result.perGradeProbs[g],
            isPredicted: g == result.grade,
          ),
        const SizedBox(height: 6),
        Row(
          children: [
            Container(width: 10, height: 10, color: GradeVueTheme.gradeColor(2)),
            const SizedBox(width: 6),
            const Expanded(
              child: Text(
                'Grades 2–4 (Moderate+) are referable.',
                style: TextStyle(
                  color: GradeVueTheme.onSurfaceMuted,
                  fontSize: 11,
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }
}

class _GradeRow extends StatelessWidget {
  const _GradeRow({
    required this.grade,
    required this.prob,
    required this.isPredicted,
  });

  final int grade;
  final double prob;
  final bool isPredicted;

  @override
  Widget build(BuildContext context) {
    final color = GradeVueTheme.gradeColor(grade);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          SizedBox(
            width: 92,
            child: Text(
              '$grade ${GradingResult.gradeLabels[grade]}',
              style: TextStyle(
                color: isPredicted ? Colors.white : GradeVueTheme.onSurfaceMuted,
                fontSize: 12,
                fontWeight: isPredicted ? FontWeight.w800 : FontWeight.w500,
              ),
            ),
          ),
          Expanded(
            child: LayoutBuilder(
              builder: (context, constraints) {
                return Stack(
                  children: [
                    Container(
                      height: 14,
                      decoration: BoxDecoration(
                        color: GradeVueTheme.surfaceAlt,
                        borderRadius: BorderRadius.circular(7),
                      ),
                    ),
                    FractionallySizedBox(
                      widthFactor: prob.clamp(0.0, 1.0),
                      child: Container(
                        height: 14,
                        decoration: BoxDecoration(
                          color: color.withValues(alpha: isPredicted ? 1.0 : 0.55),
                          borderRadius: BorderRadius.circular(7),
                          border: isPredicted
                              ? Border.all(color: Colors.white70, width: 1)
                              : null,
                        ),
                      ),
                    ),
                  ],
                );
              },
            ),
          ),
          const SizedBox(width: 8),
          SizedBox(
            width: 44,
            child: Text(
              '${(prob * 100).toStringAsFixed(1)}%',
              textAlign: TextAlign.right,
              style: TextStyle(
                color: isPredicted ? Colors.white : GradeVueTheme.onSurfaceMuted,
                fontSize: 12,
                fontWeight: isPredicted ? FontWeight.w700 : FontWeight.w500,
                fontFeatures: const [FontFeature.tabularFigures()],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
