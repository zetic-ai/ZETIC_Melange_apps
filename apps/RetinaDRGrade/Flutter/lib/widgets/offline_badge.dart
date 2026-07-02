import 'package:flutter/material.dart';

import '../theme.dart';

/// The product's whole pitch: fully offline, on-device, the image never leaves
/// the device (zero uploads).
class OfflineBadge extends StatelessWidget {
  const OfflineBadge({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: GradeVueTheme.primary.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(20),
        border:
            Border.all(color: GradeVueTheme.primary.withValues(alpha: 0.5)),
      ),
      child: const Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.wifi_off_rounded, size: 15, color: GradeVueTheme.primary),
          SizedBox(width: 6),
          Text(
            'On-device · offline · image never leaves the device',
            style: TextStyle(
              color: GradeVueTheme.primary,
              fontSize: 11.5,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}
