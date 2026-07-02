import 'package:flutter/material.dart';

import '../models/grading_result.dart';
import '../theme.dart';

/// The primary output: a large "Grade N — Label" banner, colored by severity.
class GradeBanner extends StatelessWidget {
  const GradeBanner({super.key, required this.result});

  final GradingResult result;

  @override
  Widget build(BuildContext context) {
    final color = GradeVueTheme.gradeColor(result.grade);
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 18),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.14),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color, width: 1.5),
      ),
      child: Row(
        children: [
          // Big grade numeral chip.
          Container(
            width: 56,
            height: 56,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.22),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: color, width: 1.5),
            ),
            child: Text(
              '${result.grade}',
              style: TextStyle(
                color: color,
                fontSize: 30,
                fontWeight: FontWeight.w900,
              ),
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  result.gradeHeadline,
                  style: TextStyle(
                    color: color,
                    fontSize: 24,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.3,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  'Confidence ${(result.topConfidence * 100).toStringAsFixed(1)}%',
                  style: const TextStyle(
                    color: GradeVueTheme.onSurfaceMuted,
                    fontSize: 14,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
