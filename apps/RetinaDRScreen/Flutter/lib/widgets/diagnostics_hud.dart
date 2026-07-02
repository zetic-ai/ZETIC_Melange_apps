import 'package:flutter/material.dart';

import '../models/screening_result.dart';
import '../services/preprocessor.dart';
import '../theme.dart';

/// On-screen diagnostics. In a release device build Dart `print`/`debugPrint`
/// does NOT reliably reach the native console, so per-inference latency, the two
/// raw logits, and the input tensor shape are surfaced here on the UI/HUD
/// (CLAUDE.md §5).
class DiagnosticsHud extends StatelessWidget {
  const DiagnosticsHud({
    super.key,
    required this.result,
    required this.preprocessMs,
    required this.inferenceMs,
  });

  final ScreeningResult result;
  final double preprocessMs;
  final double inferenceMs;

  @override
  Widget build(BuildContext context) {
    final shape = Preprocessor.tensorShape.join('×');
    final logits = result.logits.map((l) => l.toStringAsFixed(3)).join(', ');
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: FundusTheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: FundusTheme.surfaceAlt),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'DIAGNOSTICS',
            style: TextStyle(
              color: FundusTheme.onSurfaceMuted,
              fontSize: 11,
              fontWeight: FontWeight.w700,
              letterSpacing: 1.2,
            ),
          ),
          const SizedBox(height: 8),
          _row('Inference', '${inferenceMs.toStringAsFixed(1)} ms'),
          _row('Preprocess (Dart)', '${preprocessMs.toStringAsFixed(1)} ms'),
          _row('Input tensor', 'float32[$shape] NCHW RGB'),
          _row('Raw logits [Nrdr, Rdr]', '[$logits]'),
        ],
      ),
    );
  }

  Widget _row(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            flex: 5,
            child: Text(
              label,
              style: const TextStyle(
                color: FundusTheme.onSurfaceMuted,
                fontSize: 12,
              ),
            ),
          ),
          Expanded(
            flex: 6,
            child: Text(
              value,
              textAlign: TextAlign.right,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 12,
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
