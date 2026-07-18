/// LATE-BINDING model constants — the ONLY file in the app that names the
/// registered Melange models. Nothing else may reference these values, so
/// plugging in the GATE-0 paste-back is a one-file, one-commit change.
///
/// SnapSeek is a TWO-model pipeline, so this file owns BOTH models:
///   1. DETECT — YOLO11n salient-object localizer
///   2. EMBED  — MobileCLIP2-S0 image tower (512-d, L2-normalized in-graph)
///
/// Names are FULLY-QUALIFIED `account/project` WITH a slash (CLAUDE.md §5); a
/// bare project name throws MlangeException(3) on-device at load. The dashboard
/// does not echo a version — first upload = version 1, confirmed at create().
///
/// Values below are PLACEHOLDERS proposed in melange_upload.md. They are the
/// only fields still `[LATE-BINDING — placeholder until GATE-0 paste-back]`:
/// confirm the registered names/versions from the dashboard and overwrite here.
/// (Registration can rename the proposal — it did twice on prior apps — but a
/// mismatch is just a one-constant edit.)
class ModelRegistry {
  const ModelRegistry._();

  // --- Model 1: DETECT (YOLO11n) ------------------------------------------
  // [LATE-BINDING — placeholder until GATE-0 paste-back] (proposed)
  static const String detectName = 'ajayshah/VisualSearchDetect';
  // [LATE-BINDING — placeholder until GATE-0 paste-back] (first upload = 1)
  static const int detectVersion = 1;

  // --- Model 2: EMBED (MobileCLIP2-S0 image tower) ------------------------
  // [LATE-BINDING — placeholder until GATE-0 paste-back] (proposed)
  static const String embedName = 'ajayshah/VisualSearchEmbed';
  // [LATE-BINDING — placeholder until GATE-0 paste-back] (first upload = 1)
  static const int embedVersion = 1;
}
