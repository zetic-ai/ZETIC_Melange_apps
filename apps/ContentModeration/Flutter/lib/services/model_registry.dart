/// LATE-BINDING model constants — the SINGLE source of truth for the registered
/// Melange model name and version.
///
/// This is the ONLY file in the app that may reference the Melange model name or
/// version. Everything else (the pipeline, tests, UI, benchmarks) is buildable
/// with zero model bytes, because Melange downloads the model at runtime.
///
/// Until the GATE-0 dashboard paste-back confirms them, both values below are
/// clearly-marked PLACEHOLDERS. Plugging in the registered model is then a
/// one-file change plus one commit — nothing else needs to move.
class ModelRegistry {
  const ModelRegistry._();

  /// Fully-qualified Melange model name: `<account>/<project>` WITH the slash.
  /// A bare project name throws `MlangeException(3)` on-device at model load.
  ///
  /// [LATE-BINDING — placeholder until GATE-0 paste-back]
  static const String modelName = 'ajayshah/ContentModeration';

  /// Registered version. The dashboard does not echo a version; the first upload
  /// is version 1, confirmed at the first SDK `create()`.
  ///
  /// [LATE-BINDING — placeholder until GATE-0 paste-back]
  static const int modelVersion = 1;
}
