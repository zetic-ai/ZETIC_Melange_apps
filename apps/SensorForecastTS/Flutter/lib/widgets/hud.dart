import 'package:flutter/material.dart';

/// On-screen diagnostics. BINDING (CLAUDE.md section 5): Dart print does not
/// reach the device console in release builds, so anything worth seeing
/// during the demo/device run lives HERE, not in logs.
class Hud extends StatelessWidget {
  const Hud({
    super.key,
    required this.runMs,
    required this.pipelineUs,
    required this.score,
    required this.threshold,
    required this.eventCount,
    required this.samplesPerSecond,
    required this.modelLine,
  });

  final double runMs;
  final double pipelineUs;
  final double? score;
  final double threshold;
  final int eventCount;
  final double samplesPerSecond;
  final String modelLine;

  @override
  Widget build(BuildContext context) {
    final scoreText = score == null ? '--' : score!.toStringAsFixed(2);
    final hot = score != null && score! >= threshold;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xCC101826),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0xFF243450)),
      ),
      child: DefaultTextStyle(
        style: const TextStyle(
          fontFamily: 'monospace',
          fontSize: 12,
          color: Color(0xFFB9C6D8),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text('infer ${runMs.toStringAsFixed(1)} ms   '
                'dart ${(pipelineUs / 1000).toStringAsFixed(2)} ms   '
                '${samplesPerSecond.toStringAsFixed(0)} sps'),
            Text(
              'score $scoreText / thr ${threshold.toStringAsFixed(1)}   '
              'events $eventCount',
              style: TextStyle(
                color: hot ? const Color(0xFFFF5470) : const Color(0xFFB9C6D8),
                fontWeight: hot ? FontWeight.bold : FontWeight.normal,
              ),
            ),
            Text(modelLine, style: const TextStyle(fontSize: 10)),
          ],
        ),
      ),
    );
  }
}
