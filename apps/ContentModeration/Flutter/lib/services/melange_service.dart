import 'dart:typed_data';

import 'package:zetic_mlange/zetic_mlange.dart';

import '../config/secrets.dart';
import '../models/moderation_result.dart';
import 'model_registry.dart';
import 'postprocessor.dart';
import 'preprocessor.dart';

/// One measured inference: the moderation result plus how long `model.run` took.
class InferenceOutcome {
  const InferenceOutcome({required this.result, required this.inferenceMicros});

  final ModerationResult result;
  final int inferenceMicros;

  double get inferenceMs => inferenceMicros / 1000.0;
}

/// Owns the Melange model lifecycle for the SafeLens content gate:
/// `create` -> warm-up -> `run` -> `close`.
///
/// One-shot inference (single-image pick, no per-frame loop). The SDK binds the
/// model handle to the isolate that created it, so [load], [infer], and [close]
/// must all run on the same isolate — here, the UI isolate. Image decode +
/// resampling is pushed off-isolate by the caller (`compute`); only the small
/// `float32[1,3,384,384]` result crosses back for [infer].
///
/// The registered Melange name/version come from [ModelRegistry] — the single
/// late-binding constants file. Nothing else in the app references them.
class MelangeService {
  MelangeService({this.postprocessor = const Postprocessor()});

  final Postprocessor postprocessor;

  ZeticMLangeModel? _model;

  bool get isReady => _model != null;

  /// Download (first launch) + initialize the model, then warm it once.
  ///
  /// [onProgress] reports download progress in [0, 1] for the loading screen.
  /// modelMode RUN_AUTO: this is a ViT; the iOS-26 MPSGraph GPU crash (attention
  /// fusion) is not client-avoidable — it is handled server-side by ZETIC
  /// filtering the GPU candidate. Read the SERVED target+apType from the native
  /// console on device (see HANDOFF Tier C).
  Future<void> load({void Function(double progress)? onProgress}) async {
    if (_model != null) return;
    final model = await ZeticMLangeModel.create(
      personalKey: zeticPersonalKey,
      name: ModelRegistry.modelName,
      version: ModelRegistry.modelVersion,
      modelMode: ModelMode.runAuto,
      onProgress: onProgress,
    );
    _model = model;
    _warmUp(model);
  }

  /// One dummy inference so the first real moderation is not the slow (cold) one.
  void _warmUp(ZeticMLangeModel model) {
    final dummy = Float32List(Preprocessor.tensorLength);
    model.run([Tensor.float32List(dummy, shape: Preprocessor.tensorShape)]);
  }

  /// Run inference on already-preprocessed `float32[1,3,384,384]` input data.
  InferenceOutcome infer(Float32List inputData) {
    final model = _model;
    if (model == null) {
      throw StateError('MelangeService.infer called before load().');
    }
    final input = Tensor.float32List(inputData, shape: Preprocessor.tensorShape);
    final watch = Stopwatch()..start();
    final outputs = model.run([input]);
    watch.stop();

    final raw = outputs.first.asFloat32List();
    final result = postprocessor.classify([raw[0], raw[1]]);
    return InferenceOutcome(
      result: result,
      inferenceMicros: watch.elapsedMicroseconds,
    );
  }

  void close() {
    _model?.close();
    _model = null;
  }
}
