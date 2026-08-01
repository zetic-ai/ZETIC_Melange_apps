import 'dart:typed_data';

import 'package:zetic_mlange/zetic_mlange.dart';

import 'ctc_decoder.dart';
import 'detector_preprocessor.dart';
import 'recognizer_preprocessor.dart';

/// Wraps the ZETIC Melange lifecycle for BOTH PP-OCRv5 models.
///
/// Registered Melange models (GATE 0, Jul 2 2026) — the fully-qualified
/// `account/project` names WITH the slash are mandatory: a bare project name
/// throws `MlangeException(3)` at load on-device. The dashboard's
/// "ZETIC | ..." header is a display prefix, NOT the account.
class MelangeService {
  MelangeService();

  /// The Melange personal key is a SECRET injected at build time:
  ///   `flutter build ... --dart-define=MELANGE_PERSONAL_KEY=<key>`
  /// It is never hardcoded or committed anywhere in this repo.
  static const String _personalKey =
      String.fromEnvironment('MELANGE_PERSONAL_KEY');

  static const String _detectorName = 'ajayshah/LiveDocRedact_Detect';
  static const String _recognizerName = 'ajayshah/LiveDocRedact_Recognize';

  /// First upload (Jul 2 2026) — v1. If create() rejects the version, that is
  /// a registration mismatch to resolve with the dashboard, not a reason to
  /// guess v2 (GATE-2 ruling).
  static const int _modelVersion = 1;

  ZeticMLangeModel? _detector;
  ZeticMLangeModel? _recognizer;

  bool get isReady =>
      _detector != null &&
      !_detector!.isClosed &&
      _recognizer != null &&
      !_recognizer!.isClosed;

  /// Downloads (if needed) and initializes both models, then warms each with
  /// one dummy inference so the first camera frame is not the slow one.
  ///
  /// Backend note (CLAUDE.md §5): modelMode RUN_AUTO; the served artifact
  /// (target + apType on the native console) is ground truth — expect the CPU
  /// fallback (~129 ms det / ~32 ms rec) until `runtimeApType=NPU` is
  /// confirmed on-device.
  Future<void> init({
    void Function(double progress)? onDetectorProgress,
    void Function(double progress)? onRecognizerProgress,
  }) async {
    if (isReady) return;
    if (_personalKey.isEmpty) {
      throw StateError(
          'MELANGE_PERSONAL_KEY is missing. Build with '
          '--dart-define=MELANGE_PERSONAL_KEY=<your key> (the key is a secret '
          'supplied at build time; it is never committed).');
    }

    _detector = await ZeticMLangeModel.create(
      personalKey: _personalKey,
      name: _detectorName,
      version: _modelVersion,
      modelMode: ModelMode.runAuto,
      onProgress: onDetectorProgress,
    );
    _recognizer = await ZeticMLangeModel.create(
      personalKey: _personalKey,
      name: _recognizerName,
      version: _modelVersion,
      modelMode: ModelMode.runAuto,
      onProgress: onRecognizerProgress,
    );

    // Warm-up: one dummy inference per model.
    runDetector(Float32List(3 * kDetInputSize * kDetInputSize));
    runRecognizer(Float32List(kRecTensorLength));
  }

  /// Runs the detector on a flattened [1,3,640,640] input and returns a COPY
  /// of the [1,1,640,640] heatmap (asFloat32List is a view over a reused
  /// native buffer).
  Float32List runDetector(Float32List input) {
    final model = _requireModel(_detector, 'detector');
    final outputs = model.run([
      Tensor.float32List(input,
          shape: const [1, 3, kDetInputSize, kDetInputSize]),
    ]);
    return Float32List.fromList(outputs.first.asFloat32List());
  }

  /// Runs the recognizer on a flattened [1,3,48,320] crop and returns a COPY
  /// of the [1,40,438] probability tensor.
  Float32List runRecognizer(Float32List input) {
    final model = _requireModel(_recognizer, 'recognizer');
    final outputs = model.run([
      Tensor.float32List(input, shape: const [1, 3, kRecHeight, kRecWidth]),
    ]);
    final raw = outputs.first.asFloat32List();
    assert(raw.length == kCtcSteps * kCtcClasses);
    return Float32List.fromList(raw);
  }

  ZeticMLangeModel _requireModel(ZeticMLangeModel? model, String which) {
    if (model == null || model.isClosed) {
      throw StateError('MelangeService.$which used before init()');
    }
    return model;
  }

  void dispose() {
    if (_detector != null && !_detector!.isClosed) _detector!.close();
    if (_recognizer != null && !_recognizer!.isClosed) _recognizer!.close();
    _detector = null;
    _recognizer = null;
  }
}
