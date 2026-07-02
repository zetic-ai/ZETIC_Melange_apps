import 'dart:typed_data';

import 'package:zetic_mlange/zetic_mlange.dart';

import '../config/secrets.dart';
import '../models/screening_result.dart';
import 'postprocessor.dart';
import 'preprocessor.dart';

/// One measured inference: the verdict plus how long `model.run` took.
class InferenceOutcome {
  const InferenceOutcome({required this.result, required this.inferenceMicros});

  final ScreeningResult result;
  final int inferenceMicros;

  double get inferenceMs => inferenceMicros / 1000.0;
}

/// Owns the Melange model lifecycle for the DR screener:
/// `create` -> warm-up -> `run` -> `close`.
///
/// One-shot inference (still-image upload, no per-frame loop). The SDK binds the
/// model handle to the isolate that created it, so [load], [infer], and [close]
/// must all be called from the same isolate — here, the UI isolate. Image
/// decode/preprocess is pushed off-isolate by the caller; only the small
/// `float32[1,3,224,224]` result crosses back for [infer].
class MelangeService {
  MelangeService({this.postprocessor = const Postprocessor()});

  /// EXACT registered Melange name — must include account name and project name
  /// separated by a slash: `<account>/<project>`.
  static const String modelName = 'ajayshah/RetinaDRScreen';
  static const int modelVersion = 1;

  final Postprocessor postprocessor;

  ZeticMLangeModel? _model;

  bool get isReady => _model != null;

  /// Download (first launch) + initialize the model, then warm it once.
  ///
  /// [onProgress] reports download progress in [0, 1] for the loading screen.
  Future<void> load({void Function(double progress)? onProgress}) async {
    if (_model != null) return;
    final model = await ZeticMLangeModel.create(
      personalKey: zeticPersonalKey,
      name: modelName,
      version: modelVersion,
      modelMode: ModelMode.runAuto,
      onProgress: onProgress,
    );
    _model = model;
    _warmUp(model);
  }

  /// One dummy inference so the first real screening is not the slow (cold) one.
  void _warmUp(ZeticMLangeModel model) {
    final dummy = Float32List(Preprocessor.tensorLength);
    model.run([Tensor.float32List(dummy, shape: Preprocessor.tensorShape)]);
  }

  /// Run inference on already-preprocessed `float32[1,3,224,224]` input data.
  InferenceOutcome infer(Float32List inputData) {
    final model = _model;
    if (model == null) {
      throw StateError('MelangeService.infer called before load().');
    }
    final input = Tensor.float32List(
      inputData,
      shape: Preprocessor.tensorShape,
    );
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
