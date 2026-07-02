import 'package:flutter/material.dart';

import '../models/grading_result.dart';
import '../theme.dart';

/// A clear REFERABLE / NOT-REFERABLE flag (grade >= 2), visually distinct from
/// the grade readout. Referable derives from the argmax grade, not a separate
/// probability threshold.
class ReferableFlag extends StatelessWidget {
  const ReferableFlag({super.key, required this.result});

  final GradingResult result;

  @override
  Widget build(BuildContext context) {
    final color = result.referable
        ? GradeVueTheme.referable
        : GradeVueTheme.notReferable;
    final icon = result.referable
        ? Icons.report_gmailerrorred_rounded
        : Icons.verified_rounded;
    final subtitle = result.referable
        ? 'Grade ≥ 2 — refer for ophthalmology review'
        : 'Grade < 2 — no referral indicated';

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 16),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color),
      ),
      child: Row(
        children: [
          Icon(icon, color: color, size: 28),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  result.referableLabel,
                  style: TextStyle(
                    color: color,
                    fontSize: 18,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.5,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  subtitle,
                  style: const TextStyle(
                    color: GradeVueTheme.onSurfaceMuted,
                    fontSize: 12,
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
