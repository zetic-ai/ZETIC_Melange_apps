import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../models/label.dart';
import '../services/melange_service.dart';

/// Heads-up display: latency readout, per-class live counts, and an on-screen
/// debug line (buffer WxH + raw box). Diagnostics live here, NOT in Dart
/// `print`, because release-build Dart logs don't reach the native console.
class Hud extends StatelessWidget {
  const Hud({
    super.key,
    required this.detections,
    required this.timings,
    required this.bufferSize,
    required this.sensorOrientation,
  });

  final List<Detection> detections;
  final FrameTimings? timings;
  final Size bufferSize;
  final int sensorOrientation;

  @override
  Widget build(BuildContext context) {
    final Map<int, int> counts = <int, int>{};
    for (final Detection d in detections) {
      counts[d.classId] = (counts[d.classId] ?? 0) + 1;
    }
    final FrameTimings? t = timings;

    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(10),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            // Latency card.
            _card(
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: <Widget>[
                  const Icon(Icons.speed, size: 16, color: Color(0xFF34C759)),
                  const SizedBox(width: 6),
                  Text(
                    t == null
                        ? '— ms'
                        : '${t.totalMs.toStringAsFixed(0)} ms  '
                            '(pre ${t.preprocessMs.toStringAsFixed(0)} · '
                            'run ${t.runMs.toStringAsFixed(0)} · '
                            'post ${t.postprocessMs.toStringAsFixed(0)})',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                      fontFeatures: <FontFeature>[FontFeature.tabularFigures()],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 6),
            // Per-class counts.
            if (counts.isNotEmpty)
              _card(
                child: Wrap(
                  spacing: 8,
                  runSpacing: 4,
                  children: <Widget>[
                    for (final MapEntry<int, int> e in counts.entries)
                      _countChip(e.key, e.value),
                  ],
                ),
              ),
            const Spacer(),
            // Debug line (buffer WxH + first raw box) — the orientation pin.
            _card(
              child: Text(
                _debugLine(),
                style: const TextStyle(
                  color: Colors.white70,
                  fontSize: 10,
                  fontFeatures: <FontFeature>[FontFeature.tabularFigures()],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _debugLine() {
    final String buf =
        'buf=${bufferSize.width.toInt()}x${bufferSize.height.toInt()} '
        'sensor=$sensorOrientation';
    if (detections.isEmpty) return '$buf det=0';
    final Detection d = detections.first;
    return '$buf det=${detections.length} '
        'raw0=[${d.left.toStringAsFixed(0)},${d.top.toStringAsFixed(0)},'
        '${d.right.toStringAsFixed(0)},${d.bottom.toStringAsFixed(0)}]';
  }

  Widget _countChip(int classId, int count) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: colorForClass(classId).withValues(alpha: 0.85),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Text(
        '${labelForClass(classId)} $count',
        style: const TextStyle(
          color: Colors.black,
          fontSize: 11,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }

  Widget _card({required Widget child}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.55),
        borderRadius: BorderRadius.circular(10),
      ),
      child: child,
    );
  }
}
