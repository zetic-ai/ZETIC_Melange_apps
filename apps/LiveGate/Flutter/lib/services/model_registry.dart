/// Late-binding Melange model registry — the ONLY file in the app that names a
/// registered model or version (CLAUDE.md §4). Until the GATE-0 dashboard
/// paste-back these are clearly-marked placeholders; when the registered
/// name/version arrive, injecting them is a one-file, one-commit change and
/// nothing else in the app needs touching.
///
/// The personal key is deliberately NOT here — it follows the repo secrets
/// convention (`lib/config/secrets.dart`, gitignored). This file is committed,
/// so no secret ever belongs in it.
library;

/// A registered Melange model reference: the fully-qualified `account/project`
/// name WITH the slash, plus the version (first upload = 1).
class ModelRef {
  const ModelRef({required this.name, required this.version});

  /// e.g. `ajayshah/LiveGatePAD`. A bare project name throws MlangeException(3)
  /// on-device (CLAUDE.md §5).
  final String name;
  final int version;
}

/// The two Melange models LiveGate loads. Both are LATE-BINDING placeholders
/// until GATE-0.
class ModelRegistry {
  ModelRegistry._();

  /// PAD / anti-spoof (MiniFASNet-V2). in `input`[1,3,80,80] BGR [0,255] ->
  /// out `output`[1,3] logits.
  /// [LATE-BINDING — placeholder until GATE-0 paste-back] (requested: v1)
  static const ModelRef pad =
      ModelRef(name: 'ajayshah/LiveGatePAD', version: 1);

  /// FACE embedding (SFace / ArcFace). in `data`[1,3,112,112] BGR [0,255] ->
  /// out `fc1`[1,128] embedding (not L2-normalized in-graph).
  /// [LATE-BINDING — placeholder until GATE-0 paste-back] (requested: v1)
  static const ModelRef face =
      ModelRef(name: 'ajayshah/LiveGateFace', version: 1);
}
