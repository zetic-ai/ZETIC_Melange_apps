import 'dart:math' as math;
import 'dart:typed_data';

import '../models/gate_result.dart';

/// PAD liveness threshold: LIVE if softmax[class 1] >= this. (SPEC_STUB.md)
const double kLiveThreshold = 0.45;

/// FACE match threshold: MATCH if cosine of L2-normalized embeddings >= this.
/// 0.363 keeps impostor-accept low for a KYC gate (model_selection.md).
const double kMatchThreshold = 0.363;

/// The class index in the PAD [1,3] output that means LIVE. ⚠️ The model card
/// is wrong: it is index **1**, not 0. Classes 0 and 2 are the two spoof types.
const int kLiveClassIndex = 1;

/// Numerically-stable softmax over a small logit vector.
List<double> softmax(List<double> logits) {
  var maxLogit = logits[0];
  for (final v in logits) {
    if (v > maxLogit) maxLogit = v;
  }
  var sum = 0.0;
  final exps = List<double>.filled(logits.length, 0);
  for (var i = 0; i < logits.length; i++) {
    final e = math.exp(logits[i] - maxLogit);
    exps[i] = e;
    sum += e;
  }
  for (var i = 0; i < exps.length; i++) {
    exps[i] /= sum;
  }
  return exps;
}

/// PAD liveness score = softmax over the 3 logits, class 1.
double livenessScore(List<double> padLogits) => softmax(padLogits)[kLiveClassIndex];

/// Returns a new L2-normalized copy of [v]. A zero vector is returned unchanged
/// (its norm is 0), which downstream cosine treats as a non-match.
Float32List l2normalize(Float32List v) {
  var sumSq = 0.0;
  for (final x in v) {
    sumSq += x * x;
  }
  final norm = math.sqrt(sumSq);
  final out = Float32List(v.length);
  if (norm == 0) return out;
  for (var i = 0; i < v.length; i++) {
    out[i] = v[i] / norm;
  }
  return out;
}

/// Cosine similarity of two **already L2-normalized** vectors (a plain dot
/// product). Enrollment stores the normalized reference, and the probe is
/// normalized before this call, so both are unit vectors.
double cosine(Float32List a, Float32List b) {
  assert(a.length == b.length);
  var dot = 0.0;
  for (var i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

/// Composes the final gate verdict from the PAD logits and (optionally) the
/// probe embedding + enrolled reference.
///
/// Rules (SPEC_STUB.md): SPOOF fails the gate regardless of any match and the
/// score is withheld; LIVE + cosine >= threshold passes; LIVE + below is a real
/// but non-matching face; LIVE with no enrolled reference prompts enrollment.
///
/// [probeEmbedding] is the raw (un-normalized) SFace output; it is normalized
/// here. [enrolledNormalized] must already be L2-normalized (as stored). Pass a
/// null [probeEmbedding] to skip the match branch entirely (e.g. the caller
/// short-circuits FACE on a spoof).
GateVerdict composeVerdict({
  required List<double> padLogits,
  Float32List? probeEmbedding,
  Float32List? enrolledNormalized,
}) {
  final double live = livenessScore(padLogits);
  final bool isLive = live >= kLiveThreshold;

  if (!isLive) {
    // Spoof: fail the gate, withhold the match score entirely.
    return GateVerdict(
      decision: GateDecision.spoof,
      live: false,
      liveScore: live,
    );
  }

  if (enrolledNormalized == null || probeEmbedding == null) {
    return GateVerdict(
      decision: GateDecision.liveNoReference,
      live: true,
      liveScore: live,
    );
  }

  final cos = cosine(l2normalize(probeEmbedding), enrolledNormalized);
  return GateVerdict(
    decision:
        cos >= kMatchThreshold ? GateDecision.pass : GateDecision.liveNoMatch,
    live: true,
    liveScore: live,
    cosine: cos,
  );
}
