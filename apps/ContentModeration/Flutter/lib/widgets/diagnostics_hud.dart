import 'package:flutter/material.dart';

import '../models/moderation_result.dart';
import '../theme.dart';

/// On-screen diagnostics. In a RELEASE device build Dart `print` does NOT reach
/// the native console, so every number an engineer needs to trust the run lives
/// HERE, on the UI: per-stage latency, the decoded buffer size, the raw logits,
/// and the served-artifact reminder. (See CLAUDE.md section 5 / VALIDATION Tier C.)
class DiagnosticsHud extends StatelessWidget {
  const DiagnosticsHud({
    super.key,
    required this.result,
    required this.preprocessMs,
    required this.inferenceMs,
    required this.decodedWidth,
    required this.decodedHeight,
  });

  final ModerationResult result;
  final double preprocessMs;
  final double inferenceMs;
  final int decodedWidth;
  final int decodedHeight;

  @override
  Widget build(BuildContext context) {
    final total = preprocessMs + inferenceMs;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'DIAGNOSTICS',
              style: TextStyle(
                color: SafeLensTheme.onSurfaceMuted,
                fontSize: 11,
                fontWeight: FontWeight.w700,
                letterSpacing: 1.2,
              ),
            ),
            const SizedBox(height: 10),
            _row('Preprocess', '${preprocessMs.toStringAsFixed(1)} ms'),
            _row('Inference (model.run)', '${inferenceMs.toStringAsFixed(1)} ms'),
            _row('Total', '${total.toStringAsFixed(1)} ms'),
            const Divider(height: 18, color: SafeLensTheme.surfaceMuted),
            _row('Decoded buffer', '${decodedWidth}x$decodedHeight'),
            _row('Model input', '1x3x384x384'),
            _row(
              'Raw logits [NSFW, SFW]',
              '[${result.logits[0].toStringAsFixed(3)}, '
                  '${result.logits[1].toStringAsFixed(3)}]',
            ),
            _row('P(NSFW) / P(SFW)',
                '${result.pNsfw.toStringAsFixed(4)} / ${result.pSfw.toStringAsFixed(4)}'),
          ],
        ),
      ),
    );
  }

  Widget _row(String label, String value) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 3),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Flexible(
              child: Text(
                label,
                style: const TextStyle(
                    color: SafeLensTheme.onSurfaceMuted, fontSize: 13),
              ),
            ),
            const SizedBox(width: 12),
            Text(
              value,
              style: const TextStyle(
                color: SafeLensTheme.onSurface,
                fontSize: 13,
                fontWeight: FontWeight.w600,
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ],
        ),
      );
}
