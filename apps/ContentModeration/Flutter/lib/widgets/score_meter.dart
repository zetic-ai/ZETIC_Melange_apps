import 'package:flutter/material.dart';

import '../models/moderation_result.dart';
import '../services/postprocessor.dart';
import '../theme.dart';

/// A live P(NSFW) meter with the KEEP/REVIEW/BLOCK band boundaries marked, plus
/// the P(NSFW) / P(SFW) numbers.
class ScoreMeter extends StatelessWidget {
  const ScoreMeter({
    super.key,
    required this.result,
    this.reviewThreshold = Postprocessor.defaultReviewThreshold,
    this.blockThreshold = Postprocessor.defaultBlockThreshold,
  });

  final ModerationResult result;
  final double reviewThreshold;
  final double blockThreshold;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            _scorePill('P(NSFW)', result.pNsfw, result.decision.color),
            _scorePill('P(SFW)', result.pSfw, SafeLensTheme.onSurfaceMuted),
          ],
        ),
        const SizedBox(height: 14),
        LayoutBuilder(
          builder: (context, constraints) {
            final w = constraints.maxWidth;
            return SizedBox(
              height: 20,
              child: Stack(
                clipBehavior: Clip.none,
                children: [
                  // Three colored band segments.
                  Row(
                    children: [
                      _band(reviewThreshold, Decision.keep.color, w),
                      _band(blockThreshold - reviewThreshold,
                          Decision.review.color, w),
                      _band(1.0 - blockThreshold, Decision.block.color, w),
                    ],
                  ),
                  // The P(NSFW) marker.
                  Positioned(
                    left: (result.pNsfw.clamp(0.0, 1.0) * w) - 2,
                    child: Container(
                      width: 4,
                      height: 20,
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(2),
                        boxShadow: const [
                          BoxShadow(color: Colors.black54, blurRadius: 4),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            );
          },
        ),
        const SizedBox(height: 6),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            _tick('KEEP'),
            _tick('${(reviewThreshold * 100).toStringAsFixed(0)}%'),
            _tick('${(blockThreshold * 100).toStringAsFixed(0)}%'),
            _tick('BLOCK'),
          ],
        ),
      ],
    );
  }

  Widget _band(double fraction, Color color, double totalWidth) => Container(
        width: fraction * totalWidth,
        height: 20,
        color: color.withValues(alpha: 0.55),
      );

  Widget _tick(String label) => Text(
        label,
        style: const TextStyle(
          color: SafeLensTheme.onSurfaceMuted,
          fontSize: 10,
          fontWeight: FontWeight.w600,
        ),
      );

  Widget _scorePill(String label, double value, Color color) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label,
              style: const TextStyle(
                  color: SafeLensTheme.onSurfaceMuted, fontSize: 12)),
          const SizedBox(height: 2),
          Text(
            '${(value * 100).toStringAsFixed(1)}%',
            style: TextStyle(
              color: color,
              fontSize: 20,
              fontWeight: FontWeight.w700,
              fontFeatures: const [FontFeature.tabularFigures()],
            ),
          ),
        ],
      );
}
