import 'package:flutter/material.dart';

import '../models/screening_result.dart';
import '../theme.dart';

/// The primary output: a large REFERABLE / NOT-REFERABLE banner.
class VerdictBanner extends StatelessWidget {
  const VerdictBanner({super.key, required this.result});

  final ScreeningResult result;

  @override
  Widget build(BuildContext context) {
    final color = result.referable
        ? FundusTheme.referable
        : FundusTheme.notReferable;
    final icon = result.referable
        ? Icons.report_gmailerrorred_rounded
        : Icons.verified_rounded;

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
          Icon(icon, color: color, size: 40),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  result.verdictLabel,
                  style: TextStyle(
                    color: color,
                    fontSize: 26,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.5,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  'Confidence ${(result.confidence * 100).toStringAsFixed(1)}%',
                  style: const TextStyle(
                    color: FundusTheme.onSurfaceMuted,
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
