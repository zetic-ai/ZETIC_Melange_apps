import 'package:flutter/material.dart';

import '../theme.dart';

/// All live diagnostics for one HUD refresh. Release builds cannot log from
/// Dart (native console only shows native logs), so this HUD is the ONLY
/// observability on device — it carries every SPEC-mandated readout plus the
/// detector heatmap stats (the dashboard accuracy-anomaly check).
class HudStats {
  const HudStats({
    this.detectorMs = 0,
    this.detectorEmaMs = 0,
    this.recognizerMsPerCrop = 0,
    this.cropsThisFrame = 0,
    this.regionsRead = 0,
    this.fps = 0,
    this.droppedFrames = 0,
    this.bufferWidth = 0,
    this.bufferHeight = 0,
    this.heatmapMin = 0,
    this.heatmapMax = 0,
    this.heatmapMean = 0,
  });

  final int detectorMs;
  final double detectorEmaMs;
  final double recognizerMsPerCrop;
  final int cropsThisFrame;
  final int regionsRead;
  final double fps;
  final int droppedFrames;
  final int bufferWidth;
  final int bufferHeight;
  final double heatmapMin;
  final double heatmapMax;
  final double heatmapMean;
}

/// Top HUD: offline badge (the demo's whole pitch), latency readouts, and a
/// diagnostics line (buffer WxH + heatmap min/max/mean).
class HudBar extends StatelessWidget {
  const HudBar({super.key, required this.stats});

  final HudStats stats;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      bottom: false,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(10, 6, 10, 0),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                _brand(),
                const Spacer(),
                _offlineBadge(),
              ],
            ),
            const SizedBox(height: 6),
            Wrap(
              spacing: 6,
              runSpacing: 4,
              children: [
                _chip('DET ${stats.detectorMs} ms',
                    sub: 'ema ${stats.detectorEmaMs.toStringAsFixed(0)}'),
                _chip(
                    'REC ${stats.recognizerMsPerCrop.toStringAsFixed(0)} ms/crop'),
                _chip('CROPS ${stats.cropsThisFrame}'),
                _chip('READ ${stats.regionsRead}'),
                _chip('${stats.fps.toStringAsFixed(0)} fps',
                    sub: 'drop ${stats.droppedFrames}'),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              'buf ${stats.bufferWidth}x${stats.bufferHeight}  '
              'heat ${stats.heatmapMin.toStringAsFixed(2)}/'
              '${stats.heatmapMax.toStringAsFixed(2)}/'
              '${stats.heatmapMean.toStringAsFixed(3)}',
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.55),
                fontSize: 11,
                fontFeatures: const [FontFeature.tabularFigures()],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _brand() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: GlyphColors.background.withValues(alpha: 0.75),
        borderRadius: BorderRadius.circular(9),
      ),
      child: RichText(
        text: const TextSpan(
          children: [
            TextSpan(
              text: 'Glyph',
              style: TextStyle(
                color: Colors.white,
                fontSize: 17,
                fontWeight: FontWeight.w900,
              ),
            ),
            TextSpan(
              text: 'Go',
              style: TextStyle(
                color: GlyphColors.accent,
                fontSize: 17,
                fontWeight: FontWeight.w900,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _offlineBadge() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: GlyphColors.background.withValues(alpha: 0.75),
        borderRadius: BorderRadius.circular(9),
        border: Border.all(color: GlyphColors.ok.withValues(alpha: 0.6)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: const BoxDecoration(
              color: GlyphColors.ok,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 6),
          const Text(
            'OFFLINE · ON-DEVICE',
            style: TextStyle(
              color: GlyphColors.ok,
              fontSize: 11,
              fontWeight: FontWeight.w800,
              letterSpacing: 0.4,
            ),
          ),
        ],
      ),
    );
  }

  Widget _chip(String label, {String? sub}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: GlyphColors.background.withValues(alpha: 0.7),
        borderRadius: BorderRadius.circular(7),
      ),
      child: Text(
        sub == null ? label : '$label · $sub',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 11,
          fontWeight: FontWeight.w600,
          fontFeatures: [FontFeature.tabularFigures()],
        ),
      ),
    );
  }
}
