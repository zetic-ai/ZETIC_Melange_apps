import 'dart:typed_data';

import 'package:zetic_mlange/zetic_mlange.dart';

import '../config/secrets.dart';
import '../models/forecast.dart';
import 'preprocessor.dart';

/// One forecast pass: the raw quantile tensor copy plus per-stage timings.
class ForecastRunResult {
  const ForecastRunResult({required this.raw, required this.runMs});

  /// Copy of the float32[1,9,64] output (576 floats), safe to keep.
  final Float32List raw;

  /// Wall-clock milliseconds of the native model.run call.
  final double runMs;
}

/// Wraps the ZETIC Melange model lifecycle for ajayshah/SensorForecastTS.
///
/// SDK surface verified against the installed zetic_mlange 1.8.1 sources:
/// `ZeticMLangeModel.create(personalKey:, name:, version:, modelMode:,
/// onProgress:)`, synchronous `run` over a tensor list, `asFloat32List()`
/// (a VIEW over a reused native buffer — always copy), `close()`.
class MelangeService {
  MelangeService();

  /// FULLY-QUALIFIED name WITH the slash (CLAUDE.md section 5): a bare
  /// project name throws MlangeException(3) at on-device load. The dashboard
  /// header's "ZETIC |" prefix is display-only and never part of this string.
  static const String modelName = 'ajayshah/SensorForecastTS';
  static const int modelVersion = 1;

  ZeticMLangeModel? _model;

  // Pre-allocated input buffer (Tier B: no per-run allocation).
  final Float32List _input = Float32List(kContextLength);

  bool get isReady => _model != null && !(_model!.isClosed);

  /// Downloads (first launch) and initializes the model. Only `modelMode`
  /// reaches ZETIC's remote backend selector; RUN_AUTO per SPEC. The served
  /// artifact (target+apType) is decided server-side and can change between
  /// launches — surface what the HUD can see, promise nothing.
  Future<void> init({void Function(double progress)? onProgress}) async {
    if (isReady) return;
    _model = await ZeticMLangeModel.create(
      personalKey: zeticPersonalKey,
      name: modelName,
      version: modelVersion,
      modelMode: ModelMode.runAuto,
      onProgress: onProgress,
    );
  }

  /// One dummy inference so the first real forecast is not the slow one
  /// (backend warm-up / graph compile happens here, on the loading screen).
  void warmUp() {
    final model = _requireModel();
    for (var i = 0; i < kContextLength; i++) {
      _input[i] = 50.0 + (i % 7) * 0.1; // plausible non-degenerate values
    }
    model.run([
      Tensor.float32List(_input, shape: const [1, kContextLength])
    ]);
  }

  /// Runs one forecast on a FULL window snapshot.
  ///
  /// [window] fills the pre-allocated input via [SampleWindow.snapshotInto]
  /// (throws if not full — the export has no padding support). Raw values,
  /// no normalization. Runs on the main isolate by design: the tensor is 512
  /// floats and the model handle is bound to this isolate (PyroGuard lesson:
  /// per-run isolate spawns are a net tax).
  ForecastRunResult run(SampleWindow window) {
    final model = _requireModel();
    window.snapshotInto(_input);

    final sw = Stopwatch()..start();
    final outputs = model.run([
      Tensor.float32List(_input, shape: const [1, kContextLength])
    ]);
    sw.stop();

    final view = outputs.first.asFloat32List();
    if (view.length != kNumQuantiles * kHorizon) {
      throw StateError(
          'unexpected output length ${view.length}, want ${kNumQuantiles * kHorizon}');
    }
    return ForecastRunResult(
      raw: Float32List.fromList(view), // copy out of the reused native buffer
      runMs: sw.elapsedMicroseconds / 1000.0,
    );
  }

  ZeticMLangeModel _requireModel() {
    final model = _model;
    if (model == null || model.isClosed) {
      throw StateError('MelangeService used before init()');
    }
    return model;
  }

  void dispose() {
    final model = _model;
    if (model != null && !model.isClosed) {
      model.close();
    }
    _model = null;
  }
}
