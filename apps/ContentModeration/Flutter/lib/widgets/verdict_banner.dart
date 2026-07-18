import 'package:flutter/material.dart';

import '../models/moderation_result.dart';

/// The primary decision banner: KEEP / REVIEW-BLUR / BLOCK with its band color.
class VerdictBanner extends StatelessWidget {
  const VerdictBanner({super.key, required this.result});

  final ModerationResult result;

  @override
  Widget build(BuildContext context) {
    final d = result.decision;
    final icon = switch (d) {
      Decision.keep => Icons.check_circle_outline,
      Decision.review => Icons.remove_red_eye_outlined,
      Decision.block => Icons.block,
    };
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
      decoration: BoxDecoration(
        color: d.color.withValues(alpha: 0.16),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: d.color, width: 1.5),
      ),
      child: Row(
        children: [
          Icon(icon, color: d.color, size: 34),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  d.label,
                  style: TextStyle(
                    color: d.color,
                    fontSize: 22,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 0.5,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  d.action,
                  style: const TextStyle(color: Colors.white70, fontSize: 13),
                ),
              ],
            ),
          ),
          Text(
            '${(result.pNsfw * 100).toStringAsFixed(1)}%',
            style: TextStyle(
              color: d.color,
              fontSize: 24,
              fontWeight: FontWeight.w800,
              fontFeatures: const [FontFeature.tabularFigures()],
            ),
          ),
        ],
      ),
    );
  }
}
