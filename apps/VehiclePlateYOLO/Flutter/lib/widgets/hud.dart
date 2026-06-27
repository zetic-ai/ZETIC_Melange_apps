import 'package:flutter/material.dart';

import '../theme.dart';

/// Heads-up display: live plate count + inference latency, plus an on-screen
/// debug line (buffer WxH, rotation, upright image WxH). On a release device
/// build Dart logs do NOT reach the native console, so these diagnostics must
/// live on the UI — this is how the orientation/buffer geometry gets confirmed
/// on the real device.
class Hud extends StatelessWidget {
  const Hud({
    super.key,
    required this.plateCount,
    required this.latencyMs,
    required this.bufWidth,
    required this.bufHeight,
    required this.rotation,
    required this.imageWidth,
    required this.imageHeight,
  });

  final int plateCount;
  final double latencyMs;
  final int bufWidth;
  final int bufHeight;
  final int rotation;
  final int imageWidth;
  final int imageHeight;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              _chip(
                icon: Icons.pin_outlined,
                label: 'PLATES',
                value: '$plateCount',
              ),
              const SizedBox(width: 10),
              _chip(
                icon: Icons.speed,
                label: 'PIPELINE',
                value: latencyMs > 0
                    ? '${latencyMs.toStringAsFixed(1)} ms'
                    : '—',
              ),
            ],
          ),
          const SizedBox(height: 8),
          // Device-confirmation diagnostic line.
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.45),
              borderRadius: BorderRadius.circular(6),
            ),
            child: Text(
              'buf=${bufWidth}x$bufHeight  rot=$rotation°  '
              'img=${imageWidth}x$imageHeight',
              style: const TextStyle(
                color: AppTheme.textMuted,
                fontSize: 11,
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _chip({
    required IconData icon,
    required String label,
    required String value,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: AppTheme.surface.withValues(alpha: 0.85),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: AppTheme.accentSoft),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: AppTheme.accent),
          const SizedBox(width: 8),
          Text(
            label,
            style: const TextStyle(
              color: AppTheme.textMuted,
              fontSize: 11,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.5,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            value,
            style: const TextStyle(
              color: AppTheme.textPrimary,
              fontSize: 14,
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }
}
