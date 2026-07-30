import 'dart:typed_data';

import 'package:zetic_mlange/zetic_mlange.dart';

import '../config.dart';

/// Wraps BOTH ZETIC Melange model lifecycles (text detector + text
/// recognizer): create → warm-up → run → close.
///
/// Backend honesty (CLAUDE.md §5): modelMode is RUN_AUTO for both models, but
/// backend/precision selection is server-side and the mode does not steer the
/// served artifact. The SERVED target+apType read from the native device
/// console is ground truth — budget for the CPU fallback (~169 ms detector,
/// ~32 ms/crop recognizer) until the console shows NPU.
class MelangeService {
  MelangeService();

  ZeticMLangeModel? _detector;
  ZeticMLangeModel? _recognizer;

  bool get isReady =>
      _detector != null &&
      !_detector!.isClosed &&
      _recognizer != null &&
      !_recognizer!.isClosed;

  /// Downloads (first launch: TWO downloads — rehearse on conference Wi-Fi)
  /// and initializes both models, then warms EACH with one dummy inference so
  /// the first live frame is not the slow one.
  ///
  /// [onStatus] receives a human-readable stage plus overall progress 0..1
  /// (detector download 0–0.45, recognizer download 0.45–0.9, warm-up 0.9–1).
  Future<void> init({
    void Function(String stage, double progress)? onStatus,
  }) async {
    if (isReady) return;
    if (kMelangePersonalKey.isEmpty) {
      throw StateError(
        'MELANGE_PERSONAL_KEY is missing. Build with '
        '--dart-define=MELANGE_PERSONAL_KEY=<your key> '
        '(get one at https://mlange.zetic.ai).',
      );
    }

    onStatus?.call('Downloading text detector…', 0);
    _detector = await ZeticMLangeModel.create(
      personalKey: kMelangePersonalKey,
      name: kDetectorModelName,
      version: kModelVersion,
      modelMode: ModelMode.runAuto,
      onProgress: (p) =>
          onStatus?.call('Downloading text detector…', p.clamp(0, 1) * 0.45),
    );

    onStatus?.call('Downloading text recognizer…', 0.45);
    _recognizer = await ZeticMLangeModel.create(
      personalKey: kMelangePersonalKey,
      name: kRecognizerModelName,
      version: kModelVersion,
      modelMode: ModelMode.runAuto,
      onProgress: (p) => onStatus?.call(
          'Downloading text recognizer…', 0.45 + p.clamp(0, 1) * 0.45),
    );

    // Warm-up: one dummy inference each, so backend compilation/first-run
    // cost is paid here on the loading screen, not on the first live frame.
    onStatus?.call('Warming up detector…', 0.9);
    runDetector(Float32List(3 * kDetInputSize * kDetInputSize));
    onStatus?.call('Warming up recognizer…', 0.95);
    runRecognizer(Float32List(3 * kRecHeight * kRecWidth));
    onStatus?.call('Ready', 1);
  }

  /// Runs the detector on an NCHW [1,3,736,736] input and returns an OWNED
  /// copy of the [1,1,736,736] probability map.
  ///
  /// The copy is structural, not optional: `asFloat32List()` is a view over
  /// a REUSED native buffer, so returning it directly would let the next
  /// `run()` (either model) overwrite the caller's data. All callers get
  /// owned copies from this service; nothing outside it may touch the views.
  Float32List runDetector(Float32List input) {
    final model = _requireModel(_detector, 'detector');
    final outputs = model.run([
      Tensor.float32List(
        input,
        shape: const [1, 3, kDetInputSize, kDetInputSize],
      ),
    ]);
    return Float32List.fromList(outputs.first.asFloat32List());
  }

  /// Runs the recognizer on an NCHW [1,3,48,320] crop and returns an OWNED
  /// copy of the [1,40,838] CTC probability tensor (same copy discipline as
  /// [runDetector] — critical in the per-crop loop, where the next crop's
  /// run would overwrite the shared native view).
  Float32List runRecognizer(Float32List input) {
    final model = _requireModel(_recognizer, 'recognizer');
    final outputs = model.run([
      Tensor.float32List(input, shape: const [1, 3, kRecHeight, kRecWidth]),
    ]);
    return Float32List.fromList(outputs.first.asFloat32List());
  }

  ZeticMLangeModel _requireModel(ZeticMLangeModel? model, String name) {
    if (model == null || model.isClosed) {
      throw StateError('MelangeService: $name used before init()');
    }
    return model;
  }

  void dispose() {
    final det = _detector;
    if (det != null && !det.isClosed) det.close();
    _detector = null;
    final rec = _recognizer;
    if (rec != null && !rec.isClosed) rec.close();
    _recognizer = null;
  }
}
