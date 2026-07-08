import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:zetic_mlange/zetic_mlange.dart';

import '../models/detection.dart';
import 'postprocessor.dart';
import 'preprocessor.dart';

/// One inference pass: the detections plus the wall-clock latency that produced
/// them (preprocess + NPU run + postprocess).
class InferenceResult {
  const InferenceResult({required this.detections, required this.latencyMs});

  final List<Detection> detections;
  final int latencyMs;
}

/// Wraps the ZETIC Melange model lifecycle and the full
/// preprocess -> run -> postprocess pipeline for the fire/smoke YOLO model.
class MelangeService {
  MelangeService();

  // Get your free Melange key at https://mlange.zetic.ai and paste it here.
  static const String _personalKey = 'YOUR_MLANGE_KEY';
  static const String _modelName = 'ajayshah/FireDetectionYOLO';
  static const int _modelVersion = 1;

  ZeticMLangeModel? _model;

  bool get isReady => _model != null && !(_model!.isClosed);

  /// Downloads (if needed) and initializes the model on the NPU/GPU/CPU.
  ///
  /// [onProgress] receives download progress in 0..1 and drives the loading
  /// screen's progress bar.
  Future<void> init({void Function(double progress)? onProgress}) async {
    if (isReady) return;
    // Backend selection notes (SDK 1.8.1, verified on-device 2026-06-24):
    //   - Only `modelMode` reaches the *remote* selector (it sets the request's
    //     `selection_mode`). `target`/`apType`/`quantType` are forwarded through
    //     the FFI but ignored by mlange_model_create_remote — they only matter
    //     for local (on-path) loading. Kept here as intent/hints.
    //   - Apple's CoreML *GPU* (MPSGraph) compiler aborts this graph on
    //     iOS/macOS 26.3+ ("MLIR pass manager failed", SIGABRT) — an Apple bug.
    //     ZETIC's backend now filters the GPU candidate out for those OS
    //     versions, so we no longer crash.
    //   - With GPU filtered, every mode (AUTO/SPEED/QUANTIZED/ACCURACY)
    //     currently falls back to TFLITE_FP16 / CPU (~383ms) for
    //     `ajayshah/FireDetectionYOLO` on the A16. A CoreML / Neural-Engine
    //     artifact (~3ms) is NOT yet served. Once ZETIC serves it, runAuto
    //     should select it automatically — no app change needed.
    _model = await ZeticMLangeModel.create(
      personalKey: _personalKey,
      name: _modelName,
      version: _modelVersion,
      modelMode: ModelMode.runAuto,
      target: Target.coreMl,
      apType: APType.npu,
      onProgress: onProgress,
    );
  }

  /// Runs the full pipeline for one camera frame.
  ///
  /// Pre/post-processing (the CPU-heavy Dart) run in background isolates via
  /// [compute]; the native `model.run` runs on the main isolate because the
  /// model handle is bound to it (and the NPU pass is only a few ms).
  Future<InferenceResult> detect(
    CameraImage image,
    double confThreshold, {
    int rotationDegrees = 0,
  }) async {
    final model = _model;
    if (model == null || model.isClosed) {
      throw StateError('MelangeService.detect called before init()');
    }

    final stopwatch = Stopwatch()..start();

    // Copy planes out of the recycled camera buffer, then preprocess off-thread.
    // rotationDegrees rotates the source upright so the model sees the scene the
    // right way up (Android delivers a sensor-orientation/landscape buffer).
    final frame =
        FrameData.fromCameraImage(image, rotationDegrees: rotationDegrees);
    final pre = await compute(preprocessFrame, frame);

    final input = Tensor.float32List(
      pre.input,
      shape: const [1, 3, kInputSize, kInputSize],
    );
    final outputs = model.run([input]);
    final raw = outputs.first.asFloat32List();

    // asFloat32List() is a view over a reused native buffer; copy before it can
    // be overwritten by the next run and before crossing the isolate boundary.
    final rawCopy = Float32List.fromList(raw);

    final detections = await compute(
      postprocessOutput,
      PostprocessRequest(
        output: rawCopy,
        confThreshold: confThreshold,
        scale: pre.scale,
        padX: pre.padX,
        padY: pre.padY,
        srcWidth: pre.srcWidth,
        srcHeight: pre.srcHeight,
      ),
    );

    stopwatch.stop();
    return InferenceResult(
      detections: detections,
      latencyMs: stopwatch.elapsedMilliseconds,
    );
  }

  void dispose() {
    final model = _model;
    if (model != null && !model.isClosed) {
      model.close();
    }
    _model = null;
  }
}
