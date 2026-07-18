/// Model constants — the SINGLE source of truth for the registered Melange model
/// name and version.
///
/// This is the ONLY file in the app that may reference the Melange model name or
/// version. Everything else (the pipeline, tests, UI, benchmarks) is buildable
/// with zero model bytes, because Melange downloads the model at runtime.
///
/// GATE-0 CONFIRMED (registered 2026-07-18, tag 1d4d069225a64358b104ff9c6cbdfb0a,
/// status READY): the values below are the confirmed registered model, and the
/// served contract reconciled clean against the locally verified ONNX
/// (float32[1,3,384,384] -> float32[1,2], 22,501,083 bytes). modelMode RUN_AUTO.
class ModelRegistry {
  const ModelRegistry._();

  /// Fully-qualified Melange model name: `<account>/<project>` WITH the slash.
  /// A bare project name throws `MlangeException(3)` on-device at model load.
  ///
  /// CONFIRMED registered value (GATE-0 paste-back).
  static const String modelName = 'ajayshah/ContentModeration';

  /// Registered version (first upload = 1), CONFIRMED at the GATE-0 paste-back.
  static const int modelVersion = 1;
}
