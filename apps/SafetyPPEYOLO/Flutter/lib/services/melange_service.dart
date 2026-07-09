import 'dart:typed_data';

import 'package:zetic_mlange/zetic_mlange.dart';

import '../config/secrets.dart';
import '../models/detection.dart';
import 'postprocessor.dart';
import 'preprocessor.dart';

/// One inference pass: detections plus per-stage wall-clock latency, surfaced
/// on the HUD (Dart print does not reach the device console in release —
/// CLAUDE.md section 5).
class InferenceResult {
  const InferenceResult({
    required this.detections,
    required this.preMs,
    required this.runMs,
    required this.postMs,
    required this.bufWidth,
    required this.bufHeight,
    required this.rotationDegrees,
    required this.srcWidth,
    required this.srcHeight,
  });

  final List<Detection> detections;
  final int preMs;
  final int runMs;
  final int postMs;

  /// Raw camera buffer dimensions as delivered (debug HUD: the PyroGuard
  /// orientation bug was pinned by exactly this readout).
  final int bufWidth;
  final int bufHeight;
  final int rotationDegrees;

  /// Upright (model-facing) frame dimensions after rotation.
  final int srcWidth;
  final int srcHeight;

  int get totalMs => preMs + runMs + postMs;
}

/// Wraps the ZETIC Melange model lifecycle and the full
/// preprocess -> run -> postprocess pipeline for ajayshah/SafetyPPEYOLO.
class MelangeService {
  MelangeService();

  // The personal key lives in the GITIGNORED lib/config/secrets.dart.
  static const String _modelName = 'ajayshah/SafetyPPEYOLO';
  static const int _modelVersion = 1;

  ZeticMLangeModel? _model;
  final Preprocessor _preprocessor = Preprocessor();

  bool get isReady => _model != null && !(_model!.isClosed);

  /// Downloads (if needed) and initializes the model, then warms it with one
  /// dummy inference so the first real frame is not the slow one (Tier B).
  ///
  /// Backend-selection notes (SDK 1.8.1, PyroGuard-verified):
  ///   - Only `modelMode` reaches the remote selector; `target`/`apType` are
  ///     forwarded but ignored for remote loading — kept as intent/hints.
  ///   - The served artifact (target+apType on the native console) is the
  ///     ground truth, not the requested mode. Dashboard benchmarks NPU at
  ///     ~5.6 ms median, but PyroGuard precedent is a TFLITE_FP16/CPU serve
  ///     (~400 ms); the frame-drop guard upstream tolerates both.
  Future<void> init({void Function(double progress)? onProgress}) async {
    if (isReady) return;
    _model = await ZeticMLangeModel.create(
      personalKey: zeticPersonalKey,
      name: _modelName,
      version: _modelVersion,
      modelMode: ModelMode.runAuto,
      target: Target.coreMl,
      apType: APType.npu,
      onProgress: onProgress,
    );
    await _warmUp();
  }

  Future<void> _warmUp() async {
    final model = _model;
    if (model == null || model.isClosed) return;
    final dummy = Float32List(3 * kInputSize * kInputSize);
    final input = Tensor.float32List(
      dummy,
      shape: const [1, 3, kInputSize, kInputSize],
    );
    model.run([input]);
  }

  /// Runs the full pipeline for one still image, inline on the calling isolate.
  ///
  /// The static-image pivot only changes the INPUT SOURCE: instead of a live
  /// camera frame, [frame] wraps a decoded gallery/sample photo (upright BGRA,
  /// rotation 0). The letterbox-640 + /255 + NCHW preprocess, the channel-major
  /// [1,17,8400] decode, the class whitelist, per-class thresholds and per-class
  /// NMS are byte-for-byte identical to the original camera path.
  Future<InferenceResult> detect(FrameData frame) async {
    final model = _model;
    if (model == null || model.isClosed) {
      throw StateError('MelangeService.detect called before init()');
    }

    final sw = Stopwatch()..start();
    final pre = _preprocessor.run(frame);
    final int preMs = sw.elapsedMilliseconds;

    sw.reset();
    final input = Tensor.float32List(
      pre.input,
      shape: const [1, 3, kInputSize, kInputSize],
    );
    final outputs = model.run([input]);
    // asFloat32List() is a view over a reused native buffer; decode reads it
    // synchronously before the next run, so no defensive copy is needed here.
    final raw = outputs.first.asFloat32List();
    final int runMs = sw.elapsedMilliseconds;

    sw.reset();
    final detections = postprocessOutput(PostprocessRequest(
      output: raw,
      scale: pre.scale,
      padX: pre.padX,
      padY: pre.padY,
      srcWidth: pre.srcWidth,
      srcHeight: pre.srcHeight,
    ));
    final int postMs = sw.elapsedMilliseconds;

    return InferenceResult(
      detections: detections,
      preMs: preMs,
      runMs: runMs,
      postMs: postMs,
      bufWidth: frame.width,
      bufHeight: frame.height,
      rotationDegrees: frame.rotationDegrees,
      srcWidth: pre.srcWidth,
      srcHeight: pre.srcHeight,
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
