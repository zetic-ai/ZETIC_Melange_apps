/// The gate's verdict for one frame, and the pieces that produced it.
library;

/// The composed decision of the KYC gate.
enum GateDecision {
  /// No face detected this frame.
  noFace,

  /// PAD says SPOOF. The gate fails regardless of any match score, and the
  /// match score is deliberately NOT surfaced (see spec: do not even show the
  /// match score for a spoof).
  spoof,

  /// Face is LIVE but no reference has been enrolled yet, so no match is
  /// possible. Prompt the user to enroll.
  liveNoReference,

  /// Face is LIVE but the cosine is below threshold — a real face that is not
  /// the enrolled identity.
  liveNoMatch,

  /// Face is LIVE and matches the enrolled reference. The gate passes.
  pass,
}

/// One frame's verdict. [cosine] is null whenever a match is not applicable or
/// must be withheld (no face, spoof, or not enrolled).
class GateVerdict {
  const GateVerdict({
    required this.decision,
    required this.live,
    required this.liveScore,
    this.cosine,
  });

  const GateVerdict.noFace()
      : decision = GateDecision.noFace,
        live = false,
        liveScore = 0,
        cosine = null;

  final GateDecision decision;

  /// Whether PAD classified the crop as live (softmax[1] >= threshold).
  final bool live;

  /// PAD liveness score = softmax over the 3 logits, class 1.
  final double liveScore;

  /// Cosine of the L2-normalized probe vs the enrolled reference, or null when
  /// not applicable / withheld.
  final double? cosine;

  bool get passed => decision == GateDecision.pass;
}
