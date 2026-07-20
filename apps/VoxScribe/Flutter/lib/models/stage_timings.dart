/// Per-stage wall-clock timings for the on-screen HUD (CLAUDE.md §5: Dart
/// `print` won't surface on a release device console, so diagnostics live on
/// the HUD). All values are milliseconds unless noted.
///
/// NOTE: this is an HONEST mix — the native `run()` stages (segmentation,
/// encoder, decoder) include the NPU/CPU device time and are only meaningful
/// on a physical device; the pure-Dart stages (log-mel, powerset, detok) are
/// what the A4 micro-benchmark measures.
class StageTimings {
  StageTimings();

  double decodeWavMs = 0; // wav decode + downmix + resample
  double segPreMs = 0; // build [1,1,160000] window
  double segRunMs = 0; // segmentation model.run (device)
  double powersetMs = 0; // powerset decode + onset/offset segmentation
  int segmentsFound = 0;

  double logMelMs = 0; // total across spans (pure Dart, heavy)
  double encRunMs = 0; // total encoder run across spans (device)
  double decRunMs = 0; // total decoder run across spans (device)
  double detokMs = 0; // total detok across spans (pure Dart)

  double audioDurationSec = 0; // input clip length
  double totalMs = 0; // end-to-end pipeline wall clock

  /// Real-Time Factor: processing time / audio duration. < 1.0 is faster than
  /// real time. Device-meaningful only (includes native run stages).
  double get rtf =>
      audioDurationSec > 0 ? (totalMs / 1000.0) / audioDurationSec : 0;

  /// Pure-Dart post-processing budget (what A4 measures; device-independent).
  double get dartBudgetMs => logMelMs + powersetMs + detokMs;

  /// On-device debug line: segmentation input level + raw output stats +
  /// argmax class histogram. Compared against the offline reference to localize
  /// a 0-segments failure (input-silence vs served-artifact). Shown on the HUD.
  String diag = '';
}
