import 'package:flutter/material.dart';

/// The three content-safety decision bands, driven by score bands on P(NSFW).
///
/// Defaults (tunable in the UI): KEEP < 0.30 <= REVIEW/BLUR < 0.70 <= BLOCK.
enum Decision {
  /// Auto-approve: safe to upload as-is.
  keep('KEEP', 'Auto-approve', Color(0xFF1B9E5A)),

  /// Soft action: blur / send to human review.
  review('REVIEW / BLUR', 'Flag for review', Color(0xFFE0A100)),

  /// Hard action: do not upload.
  block('BLOCK', 'Do not upload', Color(0xFFD2382C));

  const Decision(this.label, this.action, this.color);

  /// Short band label shown on the verdict banner.
  final String label;

  /// One-line action description.
  final String action;

  /// Band color (also used to tint the blur-preview overlay).
  final Color color;

  /// Whether the picked image should be blurred in the preview (REVIEW/BLOCK).
  bool get shouldBlur => this != Decision.keep;
}

/// The result of one on-device moderation inference.
@immutable
class ModerationResult {
  const ModerationResult({
    required this.pNsfw,
    required this.pSfw,
    required this.decision,
    required this.logits,
  });

  /// P(NSFW) = softmax(logits)[0]. In [0, 1].
  final double pNsfw;

  /// P(SFW) = softmax(logits)[1] == 1 - pNsfw.
  final double pSfw;

  /// The decision band for [pNsfw] under the active thresholds.
  final Decision decision;

  /// The raw `[NSFW, SFW]` logits, for the diagnostics HUD.
  final List<double> logits;
}
