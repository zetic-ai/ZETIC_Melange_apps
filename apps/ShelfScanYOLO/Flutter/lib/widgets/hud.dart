import 'package:flutter/material.dart';

import '../services/shelf_scanner.dart';
import '../theme.dart';

/// The headline "N products detected" count plus a latency line.
class DetectionHud extends StatelessWidget {
  const DetectionHud({super.key, required this.result});

  final ScanResult result;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: ShelfSenseTheme.surface.withValues(alpha: 0.92),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: ShelfSenseTheme.accentDim.withValues(alpha: 0.5)),
      ),
      child: Row(
        children: [
          Text(
            '${result.count}',
            style: const TextStyle(
              color: ShelfSenseTheme.accent,
              fontSize: 40,
              fontWeight: FontWeight.w800,
              height: 1.0,
            ),
          ),
          const SizedBox(width: 12),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text(
                'products detected',
                style: TextStyle(fontWeight: FontWeight.w600, fontSize: 15),
              ),
              const SizedBox(height: 2),
              Text(
                '${result.totalMs.toStringAsFixed(0)} ms total',
                style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.7),
                  fontSize: 12,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

/// On-screen debug/HUD line — required because release builds swallow Dart
/// `print` (CLAUDE.md section 5). Surfaces per-stage timings, tensor shape, the
/// recorded letterbox scale/pad, and the raw first detection box.
class DebugHud extends StatelessWidget {
  const DebugHud({super.key, required this.result});

  final ScanResult result;

  @override
  Widget build(BuildContext context) {
    final t = result.transform;
    final raw = result.firstRawBox;
    final rawStr = raw == null
        ? 'none'
        : '${raw.x1.toStringAsFixed(0)},${raw.y1.toStringAsFixed(0)},'
            '${raw.x2.toStringAsFixed(0)},${raw.y2.toStringAsFixed(0)}';
    final lines = <String>[
      'img=${result.imageWidth}x${result.imageHeight} '
          'in=[1,3,640,640] out=[1,5,8400]',
      'scale=${t.scale.toStringAsFixed(4)} '
          'pad=${t.padX},${t.padY} resized=${t.resizedWidth}x${t.resizedHeight}',
      'pre=${result.preprocessMs.toStringAsFixed(1)}ms '
          'inf=${result.inferenceMs.toStringAsFixed(1)}ms '
          'post=${result.postprocessMs.toStringAsFixed(1)}ms',
      'kept(preNMS)=${result.rawKept} kept(postNMS)=${result.count} '
          'raw0=[$rawStr]',
    ];
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(8),
      color: Colors.black.withValues(alpha: 0.6),
      child: Text(
        lines.join('\n'),
        style: const TextStyle(
          fontFamily: 'monospace',
          fontSize: 10.5,
          color: Color(0xFF8AF0BE),
          height: 1.35,
        ),
      ),
    );
  }
}
