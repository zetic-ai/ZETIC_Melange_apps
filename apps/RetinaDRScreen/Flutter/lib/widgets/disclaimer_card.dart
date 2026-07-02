import 'package:flutter/material.dart';

import '../theme.dart';

/// REQUIRED, always-visible non-diagnostic disclaimer (SPEC.md "UI").
///
/// This must stay on the result surface. It states the four caveats: research/
/// capability proof (not diagnostic), on-device = data-residency only, no FDA
/// clearance, and binary screen (not a severity grade / clinical diagnosis).
class DisclaimerCard extends StatelessWidget {
  const DisclaimerCard({super.key});

  static const String text =
      'Research capability demo — NOT a diagnostic device and NOT FDA-cleared. '
      'On-device inference changes data-residency / offline posture only, not '
      'clinical validity. This is a BINARY referable / not-referable screen, not '
      'a 0–4 severity grade and not a clinical diagnosis. Do not use for patient '
      'care.';

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: FundusTheme.surfaceAlt,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.amber.withValues(alpha: 0.5)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.warning_amber_rounded,
              color: Colors.amber, size: 18),
          const SizedBox(width: 8),
          const Expanded(
            child: Text(
              text,
              style: TextStyle(
                color: FundusTheme.onSurfaceMuted,
                fontSize: 11.5,
                height: 1.35,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
