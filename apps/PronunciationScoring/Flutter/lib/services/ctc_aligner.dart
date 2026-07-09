import '../models/phonemes.dart';
import 'postprocessor.dart';

/// CTC forced alignment (Viterbi) of a known target phoneme sequence over the
/// 64 CTC frames.
///
/// EXACT contract = validation/validate_onnx.py `ctc_forced_align`. The target
/// is expanded to the standard CTC lattice [blank, p1, blank, p2, ..., blank].
/// Transitions from state s: stay (s), advance (s-1), or SKIP (s-2) — the skip
/// is allowed ONLY when the current state is a real phoneme AND differs from
/// the phoneme two back (i.e. skip across a single blank between DIFFERENT
/// phonemes). A repeated phoneme therefore cannot be skipped into, forcing a
/// blank between the two copies. Log-probs are summed along the path.
class CtcAligner {
  const CtcAligner();

  /// Returns, for each target phoneme, the sorted list of frame indices the
  /// best path assigned to it. Phonemes with no aligned frames get an empty
  /// list (they will score 0 downstream).
  List<List<int>> align(LogProbView lp, List<int> targetIds) {
    final t = lp.frames;
    if (targetIds.isEmpty) return const [];

    // Expanded lattice: [blank, t0, blank, t1, ..., tn, blank].
    final ext = List<int>.filled(2 * targetIds.length + 1, Phonemes.blank);
    for (var i = 0; i < targetIds.length; i++) {
      ext[2 * i + 1] = targetIds[i];
    }
    final s = ext.length;
    const neg = -1e30;

    // dp[time][state], bp = backpointer state.
    final dp = List<List<double>>.generate(
        t, (_) => List<double>.filled(s, neg),
        growable: false);
    final bp = List<List<int>>.generate(t, (_) => List<int>.filled(s, 0),
        growable: false);

    dp[0][0] = lp.at(0, ext[0]);
    if (s > 1) dp[0][1] = lp.at(0, ext[1]);

    for (var ti = 1; ti < t; ti++) {
      final prev = dp[ti - 1];
      final cur = dp[ti];
      final curBp = bp[ti];
      for (var si = 0; si < s; si++) {
        var best = prev[si];
        var arg = si;
        if (si >= 1 && prev[si - 1] > best) {
          best = prev[si - 1];
          arg = si - 1;
        }
        if (si >= 2 &&
            ext[si] != Phonemes.blank &&
            ext[si] != ext[si - 2] &&
            prev[si - 2] > best) {
          best = prev[si - 2];
          arg = si - 2;
        }
        if (best > neg / 2) {
          cur[si] = best + lp.at(ti, ext[si]);
          curBp[si] = arg;
        }
      }
    }

    // Backtrack from the better of the two terminal states.
    var si = (s >= 2 && dp[t - 1][s - 1] >= dp[t - 1][s - 2]) ? s - 1 : s - 2;
    if (si < 0) si = 0;
    final path = List<int>.filled(t, 0);
    path[t - 1] = si;
    for (var ti = t - 1; ti > 0; ti--) {
      si = bp[ti][si];
      path[ti - 1] = si;
    }

    final frames = List<List<int>>.generate(targetIds.length, (_) => <int>[],
        growable: false);
    for (var ti = 0; ti < t; ti++) {
      final st = path[ti];
      if (ext[st] != Phonemes.blank) {
        frames[(st - 1) ~/ 2].add(ti);
      }
    }
    return frames;
  }
}
