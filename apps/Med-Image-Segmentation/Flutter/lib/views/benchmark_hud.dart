import 'package:flutter/material.dart';

import '../benchmark/benchmark_stats.dart';
import '../benchmark/device_info.dart';
import '../theme.dart';

/// The per-device benchmark panel: the table a prospect doesn't currently have.
class BenchmarkHud extends StatelessWidget {
  const BenchmarkHud({super.key, required this.snapshot, required this.device});

  final BenchmarkSnapshot snapshot;
  final DeviceIdentity? device;

  @override
  Widget build(BuildContext context) {
    final s = snapshot;

    return Panel(
      padding: const EdgeInsets.fromLTRB(18, 16, 18, 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // grid of metrics, two per row (fixed columns → values never reflow)
          _grid(s),
          if (device != null) ...[
            const SizedBox(height: 16),
            Row(
              children: [
                const Icon(Icons.smartphone, size: 12, color: Palette.textLow),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    '${device!.model}  ·  ${device!.os}',
                    style: const TextStyle(
                      color: Palette.textLow,
                      fontSize: 11.5,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }

  Widget _grid(BenchmarkSnapshot s) {
    final tiles = <Widget>[
      StatTile(
        label: 'Inference',
        value: '${s.inferenceMs.toStringAsFixed(1)} ms',
        sub: 'p95 ${s.inferenceP95Ms.toStringAsFixed(1)}',
        accent: Palette.cyan,
        icon: Icons.bolt,
      ),
      StatTile(
        label: 'Accuracy vs FP32',
        value: '${(s.meanIoU * 100).toStringAsFixed(1)}%',
        sub: 'mean IoU',
        accent: Palette.green,
        icon: Icons.verified_outlined,
      ),
      StatTile(
        label: 'Preprocess',
        value: '${s.preprocessMs.toStringAsFixed(1)} ms',
        sub: ' ',
        icon: Icons.tune,
      ),
      StatTile(
        label: 'Memory',
        value: '${s.memoryMB.toStringAsFixed(0)} MB',
        sub: 'peak ${s.peakMemoryMB.toStringAsFixed(0)}',
        icon: Icons.memory,
      ),
      StatTile(
        label: 'Total / frame',
        value: '${s.totalMs.toStringAsFixed(1)} ms',
        sub: ' ',
        icon: Icons.timelapse,
      ),
      StatTile(
        label: 'Frames',
        value: '${s.frames}',
        sub: ' ',
        icon: Icons.layers_outlined,
      ),
    ];
    return Column(
      children: [
        for (var i = 0; i < tiles.length; i += 2) ...[
          _row(
            tiles[i],
            i + 1 < tiles.length ? tiles[i + 1] : const SizedBox(),
          ),
          if (i + 2 < tiles.length) const SizedBox(height: 18),
        ],
      ],
    );
  }

  Widget _row(Widget a, Widget b) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(child: a),
        const SizedBox(width: 16),
        Expanded(child: b),
      ],
    );
  }
}
