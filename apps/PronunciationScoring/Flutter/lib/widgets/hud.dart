import 'package:flutter/material.dart';

import '../theme.dart';

/// Top HUD: SayRight wordmark + a diagnostics line (measured latency, served
/// artifact/mode, and the capture-rate note e.g. "48k→16k decimation active").
class Hud extends StatelessWidget {
  const Hud({
    super.key,
    required this.inferenceMs,
    required this.scoringMs,
    required this.servedArtifact,
    required this.sampleRateInfo,
  });

  final int inferenceMs;
  final int scoringMs;
  final String servedArtifact;
  final String? sampleRateInfo;

  @override
  Widget build(BuildContext context) {
    final hasRun = inferenceMs > 0;
    return SafeArea(
      bottom: false,
      child: Container(
        margin: const EdgeInsets.fromLTRB(12, 10, 12, 0),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: SayColors.scrim,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.record_voice_over,
                    color: SayColors.accent, size: 22),
                const SizedBox(width: 8),
                const Text('Say',
                    style: TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.w800)),
                const Text('Right',
                    style: TextStyle(
                        color: SayColors.accent,
                        fontSize: 18,
                        fontWeight: FontWeight.w800)),
                const Spacer(),
                _Pill(
                  label: hasRun ? '$inferenceMs ms ⚡' : '-- ms',
                ),
              ],
            ),
            const SizedBox(height: 6),
            Text(
              _diagnostics(hasRun),
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.5),
                fontSize: 11,
                fontFeatures: const [FontFeature.tabularFigures()],
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _diagnostics(bool hasRun) {
    final parts = <String>['served: $servedArtifact'];
    if (hasRun) parts.add('score ${scoringMs}ms');
    if (sampleRateInfo != null) parts.add(sampleRateInfo!);
    return parts.join('  •  ');
  }
}

class _Pill extends StatelessWidget {
  const _Pill({required this.label});
  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.35),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Text(
        label,
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
