import 'package:flutter/material.dart';

import '../theme.dart';

/// Bottom stats bar: per-class live counts, refreshed every frame.
class StatsBar extends StatelessWidget {
  const StatsBar({
    super.key,
    required this.fireCount,
    required this.smokeCount,
  });

  final int fireCount;
  final int smokeCount;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      top: false,
      child: Container(
        margin: const EdgeInsets.fromLTRB(12, 0, 12, 12),
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
        decoration: BoxDecoration(
          color: PyroColors.scrim,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            _StatChip(
              emoji: '🔥',
              label: 'Fire',
              value: fireCount,
              color: PyroColors.fire,
            ),
            Container(width: 1, height: 24, color: Colors.white12),
            _StatChip(
              emoji: '💨',
              label: 'Smoke',
              value: smokeCount,
              color: PyroColors.smoke,
            ),
          ],
        ),
      ),
    );
  }
}

class _StatChip extends StatelessWidget {
  const _StatChip({
    required this.emoji,
    required this.label,
    required this.value,
    required this.color,
  });

  final String emoji;
  final String label;
  final int value;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(emoji, style: const TextStyle(fontSize: 18)),
        const SizedBox(width: 8),
        Text(
          '$label: ',
          style: const TextStyle(
            color: Colors.white70,
            fontSize: 15,
            fontWeight: FontWeight.w600,
          ),
        ),
        Text(
          '$value',
          style: TextStyle(
            color: color,
            fontSize: 18,
            fontWeight: FontWeight.w800,
            fontFeatures: const [FontFeature.tabularFigures()],
          ),
        ),
      ],
    );
  }
}
