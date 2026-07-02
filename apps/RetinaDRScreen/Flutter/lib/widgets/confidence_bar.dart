import 'package:flutter/material.dart';

import '../services/postprocessor.dart';
import '../theme.dart';

/// A horizontal P(referable) probability bar with the fixed 0.5 decision
/// threshold marked. Left = not-referable, right = referable.
class ConfidenceBar extends StatelessWidget {
  const ConfidenceBar({super.key, required this.pReferable});

  final double pReferable;

  @override
  Widget build(BuildContext context) {
    final referable = pReferable >= Postprocessor.defaultThreshold;
    final fillColor = referable
        ? FundusTheme.referable
        : FundusTheme.notReferable;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text(
              'P(referable)',
              style: TextStyle(
                color: FundusTheme.onSurfaceMuted,
                fontSize: 13,
              ),
            ),
            Text(
              pReferable.toStringAsFixed(4),
              style: const TextStyle(
                color: Colors.white,
                fontSize: 14,
                fontWeight: FontWeight.w700,
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        LayoutBuilder(
          builder: (context, constraints) {
            final width = constraints.maxWidth;
            return SizedBox(
              height: 16,
              child: Stack(
                children: [
                  // Track.
                  Container(
                    height: 16,
                    decoration: BoxDecoration(
                      color: FundusTheme.surfaceAlt,
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                  // Fill up to P(referable).
                  FractionallySizedBox(
                    widthFactor: pReferable.clamp(0.0, 1.0),
                    child: Container(
                      height: 16,
                      decoration: BoxDecoration(
                        color: fillColor,
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                  ),
                  // 0.5 threshold marker.
                  Positioned(
                    left: width * Postprocessor.defaultThreshold - 1,
                    top: -2,
                    bottom: -2,
                    child: Container(width: 2, color: Colors.white70),
                  ),
                ],
              ),
            );
          },
        ),
        const SizedBox(height: 6),
        const Text(
          'Threshold fixed at 0.50 (argmax). Not adjustable in this build.',
          style: TextStyle(color: FundusTheme.onSurfaceMuted, fontSize: 11),
        ),
      ],
    );
  }
}
