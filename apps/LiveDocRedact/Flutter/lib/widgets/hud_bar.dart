import 'package:flutter/material.dart';

import '../theme.dart';

/// Per-frame pipeline timings shown on the HUD. All diagnostics live
/// ON-SCREEN because Dart print does not reach the native console in a
/// release device build (CLAUDE.md §5).
class HudTimings {
  const HudTimings({
    this.preMs = 0,
    this.detMs = 0,
    this.decodeMs = 0,
    this.recMs = 0,
    this.cropsThisFrame = 0,
    this.totalMs = 0,
  });

  final int preMs;
  final int detMs;
  final int decodeMs;
  final int recMs;
  final int cropsThisFrame;
  final int totalMs;
}

/// Top HUD: product badge, "on-device" pledge, per-stage latencies and the
/// orientation/heatmap debug line (the PyroGuard lesson — one on-screen debug
/// line is what pins orientation bugs on-device).
class HudBar extends StatelessWidget {
  const HudBar({
    super.key,
    required this.timings,
    required this.debugLine,
    required this.onSettingsTap,
  });

  final HudTimings timings;
  final String debugLine;
  final VoidCallback onSettingsTap;

  @override
  Widget build(BuildContext context) {
    final t = timings;
    return SafeArea(
      bottom: false,
      child: Container(
        margin: const EdgeInsets.fromLTRB(12, 8, 12, 0),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.55),
          borderRadius: BorderRadius.circular(14),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              children: [
                const Icon(Icons.remove_red_eye_outlined,
                    color: RedactColors.accent, size: 18),
                const SizedBox(width: 6),
                const Text(
                  'RedactLens',
                  style: TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w800,
                    fontSize: 15,
                    letterSpacing: 0.3,
                  ),
                ),
                const SizedBox(width: 10),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: RedactColors.accent.withValues(alpha: 0.18),
                    borderRadius: BorderRadius.circular(999),
                    border: Border.all(
                        color: RedactColors.accent.withValues(alpha: 0.7)),
                  ),
                  child: const Text(
                    'ON-DEVICE · NO CLOUD',
                    style: TextStyle(
                      color: RedactColors.accent,
                      fontSize: 9.5,
                      fontWeight: FontWeight.w800,
                      letterSpacing: 0.6,
                    ),
                  ),
                ),
                const Spacer(),
                GestureDetector(
                  onTap: onSettingsTap,
                  child: const Icon(Icons.tune, color: Colors.white70, size: 20),
                ),
              ],
            ),
            const SizedBox(height: 6),
            Text(
              'det ${t.detMs}ms · dbdec ${t.decodeMs}ms · '
              'rec ${t.recMs}ms×${t.cropsThisFrame} · pre ${t.preMs}ms · '
              'total ${t.totalMs}ms',
              style: const TextStyle(
                color: Colors.white70,
                fontSize: 11,
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
            const SizedBox(height: 2),
            Text(
              debugLine,
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.45),
                fontSize: 10,
                fontFeatures: const [FontFeature.tabularFigures()],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
