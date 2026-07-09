import 'package:flutter/material.dart';

import '../theme.dart';

/// Bottom stats bar: live per-class counts, worn (green) vs violation (red).
class StatsBar extends StatelessWidget {
  const StatsBar({
    super.key,
    required this.hardhatCount,
    required this.vestCount,
    required this.noHardhatCount,
    required this.noVestCount,
  });

  final int hardhatCount;
  final int vestCount;
  final int noHardhatCount;
  final int noVestCount;

  @override
  Widget build(BuildContext context) {
    final int violations = noHardhatCount + noVestCount;
    return SafeArea(
      top: false,
      child: Container(
        margin: const EdgeInsets.fromLTRB(12, 0, 12, 10),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          color: SiteColors.scrim,
          borderRadius: BorderRadius.circular(14),
          border: violations > 0
              ? Border.all(color: SiteColors.noHardhat, width: 1.5)
              : null,
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            _Stat(
              icon: Icons.hardware,
              label: 'Hardhat',
              count: hardhatCount,
              color: SiteColors.hardhat,
            ),
            _Stat(
              icon: Icons.checkroom,
              label: 'Vest',
              count: vestCount,
              color: SiteColors.vest,
            ),
            _Stat(
              icon: Icons.warning_amber_rounded,
              label: 'Violations',
              count: violations,
              color: violations > 0 ? SiteColors.noHardhat : Colors.white38,
            ),
          ],
        ),
      ),
    );
  }
}

class _Stat extends StatelessWidget {
  const _Stat({
    required this.icon,
    required this.label,
    required this.count,
    required this.color,
  });

  final IconData icon;
  final String label;
  final int count;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, color: color, size: 20),
        const SizedBox(width: 6),
        Text(
          '$count',
          style: TextStyle(
            color: color,
            fontSize: 18,
            fontWeight: FontWeight.w800,
            fontFeatures: const [FontFeature.tabularFigures()],
          ),
        ),
        const SizedBox(width: 4),
        Text(
          label,
          style: TextStyle(
            color: Colors.white.withValues(alpha: 0.65),
            fontSize: 12,
          ),
        ),
      ],
    );
  }
}
