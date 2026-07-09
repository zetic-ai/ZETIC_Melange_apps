import 'package:flutter/material.dart';

import '../theme.dart';

/// Top HUD: brand, per-stage latency readout, settings + debug toggles.
class HudBar extends StatelessWidget {
  const HudBar({
    super.key,
    required this.preMs,
    required this.runMs,
    required this.postMs,
    required this.onSettingsTap,
    required this.onDebugTap,
    required this.debugOn,
  });

  final int preMs;
  final int runMs;
  final int postMs;
  final VoidCallback onSettingsTap;
  final VoidCallback onDebugTap;
  final bool debugOn;

  @override
  Widget build(BuildContext context) {
    final int total = preMs + runMs + postMs;
    return SafeArea(
      bottom: false,
      child: Container(
        margin: const EdgeInsets.fromLTRB(12, 8, 12, 0),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: SiteColors.scrim,
          borderRadius: BorderRadius.circular(14),
        ),
        child: Row(
          children: [
            const Icon(Icons.engineering, color: SiteColors.accent, size: 22),
            const SizedBox(width: 8),
            const Text(
              'SiteGuard',
              style: TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.w800,
                fontSize: 16,
                letterSpacing: 0.4,
              ),
            ),
            const Spacer(),
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  '$total ms',
                  style: const TextStyle(
                    color: SiteColors.accent,
                    fontWeight: FontWeight.w800,
                    fontSize: 15,
                    fontFeatures: [FontFeature.tabularFigures()],
                  ),
                ),
                Text(
                  'pre $preMs · run $runMs · post $postMs',
                  style: TextStyle(
                    color: Colors.white.withValues(alpha: 0.55),
                    fontSize: 10,
                    fontFeatures: const [FontFeature.tabularFigures()],
                  ),
                ),
              ],
            ),
            const SizedBox(width: 10),
            IconButton(
              onPressed: onDebugTap,
              icon: Icon(
                Icons.bug_report_outlined,
                color: debugOn ? SiteColors.accent : Colors.white54,
                size: 22,
              ),
              tooltip: 'Debug HUD',
            ),
            IconButton(
              onPressed: onSettingsTap,
              icon: const Icon(Icons.tune, color: Colors.white70, size: 22),
              tooltip: 'Settings',
            ),
          ],
        ),
      ),
    );
  }
}
