import 'package:flutter/material.dart';

import '../services/melange_service.dart';
import '../theme.dart';

/// On-screen diagnostics (release-safe: Dart print does NOT reach the device
/// console in release builds, so per-stage latency, frame WxH and the primary
/// box coords must live on the HUD — CLAUDE.md §5). Monospace so numbers align.
class HudBar extends StatelessWidget {
  const HudBar({super.key, required this.outcome});

  final SearchOutcome outcome;

  @override
  Widget build(BuildContext context) {
    final p = outcome.primary;
    final box = p == null
        ? 'none (center-crop fallback)'
        : '[${p.rect.left.toStringAsFixed(2)},${p.rect.top.toStringAsFixed(2)}'
            ' ${p.rect.right.toStringAsFixed(2)},${p.rect.bottom.toStringAsFixed(2)}]';
    final lines = <String>[
      'detect ${outcome.detectMs}ms   embed ${outcome.embedMs}ms   '
          'total ${outcome.totalMs}ms',
      'frame ${outcome.frameWidth}x${outcome.frameHeight}   '
          'box $box',
    ];
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      color: Colors.black.withValues(alpha: 0.55),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          for (final l in lines)
            Text(l,
                style: const TextStyle(
                    color: SnapColors.accent,
                    fontFamily: 'monospace',
                    fontSize: 11)),
        ],
      ),
    );
  }
}
