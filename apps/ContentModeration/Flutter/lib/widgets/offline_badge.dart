import 'package:flutter/material.dart';

import '../theme.dart';

/// A small "100% on-device" badge — the core Trust & Safety selling point:
/// the image never leaves the phone.
class OnDeviceBadge extends StatelessWidget {
  const OnDeviceBadge({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: SafeLensTheme.surfaceMuted,
        borderRadius: BorderRadius.circular(20),
      ),
      child: const Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.lock_outline, size: 14, color: SafeLensTheme.accent),
          SizedBox(width: 6),
          Text(
            '100% on-device · no upload',
            style: TextStyle(
              color: SafeLensTheme.onSurface,
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}
