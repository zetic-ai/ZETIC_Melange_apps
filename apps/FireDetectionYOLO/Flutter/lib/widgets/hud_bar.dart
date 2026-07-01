import 'package:flutter/material.dart';

import '../theme.dart';

/// Top HUD: PyroGuard wordmark + flame on the left, live inference latency and
/// a settings button on the right.
class HudBar extends StatelessWidget {
  const HudBar({
    super.key,
    required this.latencyMs,
    required this.onSettingsTap,
  });

  final int latencyMs;
  final VoidCallback onSettingsTap;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      bottom: false,
      child: Container(
        margin: const EdgeInsets.fromLTRB(12, 10, 12, 0),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: PyroColors.scrim,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Row(
          children: [
            const Icon(Icons.local_fire_department,
                color: PyroColors.accent, size: 22),
            const SizedBox(width: 8),
            Text(
              'Pyro',
              style: const TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w800,
                letterSpacing: 0.5,
              ),
            ),
            const Text(
              'Guard',
              style: TextStyle(
                color: PyroColors.accent,
                fontSize: 18,
                fontWeight: FontWeight.w800,
                letterSpacing: 0.5,
              ),
            ),
            const Spacer(),
            _LatencyPill(latencyMs: latencyMs),
            const SizedBox(width: 6),
            InkResponse(
              onTap: onSettingsTap,
              radius: 22,
              child: const Padding(
                padding: EdgeInsets.all(4),
                child: Icon(Icons.tune, color: Colors.white70, size: 22),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _LatencyPill extends StatelessWidget {
  const _LatencyPill({required this.latencyMs});

  final int latencyMs;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.35),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Text(
        latencyMs <= 0 ? '-- ms' : '$latencyMs ms ⚡',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 13,
          fontWeight: FontWeight.w700,
          fontFeatures: [FontFeature.tabularFigures()],
        ),
      ),
    );
  }
}
