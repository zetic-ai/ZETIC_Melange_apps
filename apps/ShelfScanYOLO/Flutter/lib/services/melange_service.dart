import 'dart:typed_data';

import 'package:zetic_mlange/zetic_mlange.dart';

import '../config/secrets.dart';
import 'preprocessor.dart';

/// Lifecycle wrapper around the ZETIC Melange model:
/// create (download + compile) -> warm-up -> run -> close.
///
/// The native model handle is bound to the isolate that created it, so
/// [initialize], [runInference] and [close] must all be called from the same
/// isolate (the app runs them on the main isolate — inference is a one-shot per
/// user action, not a per-frame loop, so there is no per-frame isolate to
/// amortize).
class MelangeService {
  MelangeService();

  /// EXACT registered Melange name (string after "ZETIC |", case-sensitive).
  static const String modelName = 'ajayshah/ShelfScanYOLO';
  static const int modelVersion = 1;

  static const int _inputSize = Preprocessor.inputSize;
  static const List<int> inputShape = [1, 3, _inputSize, _inputSize];

  ZeticMLangeModel? _model;

  bool get isReady => _model != null && !_model!.isClosed;

  /// Download + prepare the model, then warm it with one dummy inference so the
  /// first real detection is not the slow (compile/cold) one.
  ///
  /// [onProgress] receives 0.0..1.0 during the model download.
  Future<void> initialize({void Function(double progress)? onProgress}) async {
    final model = await ZeticMLangeModel.create(
      personalKey: zeticPersonalKey,
      name: modelName,
      version: modelVersion,
      modelMode: ModelMode.runAuto,
      onProgress: onProgress,
    );
    _model = model;
    _warmUp();
  }

  void _warmUp() {
    final model = _model;
    if (model == null) return;
    final dummy = Float32List(3 * _inputSize * _inputSize);
    model.run([Tensor.float32List(dummy, shape: inputShape)]);
  }

  /// Run one inference. [input] is the flattened NCHW `float32[1,3,640,640]`.
  /// Returns the raw `output0` as `float32[1,5,8400]` (channel-major).
  Float32List runInference(Float32List input) {
    final model = _model;
    if (model == null) {
      throw StateError('MelangeService.initialize() has not completed.');
    }
    final outputs = model.run([Tensor.float32List(input, shape: inputShape)]);
    // Copy out of the native-backed view before the buffer can be reused.
    return Float32List.fromList(outputs.first.asFloat32List());
  }

  void close() {
    _model?.close();
    _model = null;
  }
}
