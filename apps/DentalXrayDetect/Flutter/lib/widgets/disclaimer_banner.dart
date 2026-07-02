import 'package:flutter/material.dart';

import '../theme.dart';

/// Required, always-visible non-diagnostic disclaimer.
///
/// This must be present in EVERY UI state (loading + analyzer). It states
/// plainly that OraLens is a research capability proof, NOT a diagnostic device;
/// nothing shown is clinically validated; and on-device deployment changes
/// data-residency ONLY and does not imply or confer FDA clearance.
class DisclaimerBanner extends StatelessWidget {
  const DisclaimerBanner({super.key});

  static const String text =
      'Research capability proof — NOT a diagnostic device. Nothing shown is '
      'clinically validated. On-device inference changes data-residency only '
      'and does not imply or confer FDA clearance.';

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(14, 8, 14, 10),
      decoration: const BoxDecoration(
        color: AppTheme.surface,
        border: Border(top: BorderSide(color: AppTheme.accentSoft)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: const <Widget>[
          Icon(Icons.warning_amber_rounded, size: 15, color: AppTheme.warn),
          SizedBox(width: 8),
          Expanded(
            child: Text(
              text,
              style: TextStyle(
                color: AppTheme.textMuted,
                fontSize: 10.5,
                height: 1.3,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
