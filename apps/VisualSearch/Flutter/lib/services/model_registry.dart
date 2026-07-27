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
/// GATE-0 paste-back CONFIRMED (registered 2026-07-18, uploads automated via the
/// Melange Python SDK). Both models registered and READY; registered names match
/// the proposals exactly and reconciliation was CLEAN — the dashboard confirmed
/// the stored artifacts byte-identical to the locally verified ONNX, so served
/// contracts == locally verified contracts ([1,3,640,640]→[1,84,8400];
/// [1,3,256,256]→[1,512]). modelMode RUN_AUTO.
class ModelRegistry {
  const ModelRegistry._();

  // --- Model 1: DETECT (YOLO11n) ------------------------------------------
  // CONFIRMED 2026-07-18 — READY, tag 91183404e8b248a5b1f652d48f1d5660,
  // reconciled clean (detect 10,741,317 B + sample 4,915,328 B).
  static const String detectName = 'ajayshah/VisualSearchDetect';
  static const int detectVersion = 1;

  // --- Model 2: EMBED (MobileCLIP2-S0 image tower) ------------------------
  // CONFIRMED 2026-07-18 — READY, tag 8939129b2b6848908a60e1d2e687e88b,
  // reconciled clean (embed 45,806,120 B + sample 786,560 B).
  static const String embedName = 'ajayshah/VisualSearchEmbed';
  static const int embedVersion = 1;
}
