import 'dart:typed_data';

import 'package:zetic_mlange/zetic_mlange.dart';

import 'postprocessor.dart' show DecoderStep;

/// Owns the THREE Melange model handles (segmentation + Whisper encoder +
/// Whisper decoder) for the VoxScribe pipeline.
///
/// CRITICAL (CLAUDE.md §5): the SDK binds a native model handle to the isolate
/// that called `create`, and `run()` is synchronous. So create / run / close
/// for all three models MUST happen on the SAME isolate. This object therefore
/// lives ENTIRELY inside the dedicated worker isolate (see pipeline_isolate).
///
/// modelMode = RUN_AUTO for all three (GATE-2 decision 9). Served backend is
/// chosen server-side and is not steerable from the client.
class MelangeService {
  MelangeService({required this.personalKey});

  final String personalKey;

  // Registered Melange model names (SPEC) — all version 1.
  // Segmentation: ZETIC re-triggered the conversion with the LSTM issue fixed
  // and re-registered as `ajayshah/pyannote-segmentation-3.0` — this artifact is
  // CoreML/NPU-accelerated on Apple (~9.7 ms on iPhone 15) and numerically
  // correct, unlike the earlier `ajayshah/diarization` TFLite/CPU artifact.
  static const String segName = 'ajayshah/pyannote-segmentation-3.0';
  static const String encName = 'OpenAI/whisper-tiny-encoder';
  static const String decName = 'OpenAI/whisper-tiny-decoder';

  static const List<int> segInShape = <int>[1, 1, 160000];
  static const List<int> encInShape = <int>[1, 80, 3000];
  static const List<int> encOutShape = <int>[1, 1500, 384];
  static const List<int> decTokShape = <int>[1, 448];

  ZeticMLangeModel? _seg;
  ZeticMLangeModel? _enc;
  ZeticMLangeModel? _dec;

  bool get isLoaded => _seg != null && _enc != null && _dec != null;

  /// Loads all three models (RUN_AUTO) and warms each with one dummy inference
  /// so the first real span is not the slow compile-on-first-run path.
  /// [onStage] reports (stageIndex 0..2, fractionalProgress 0..1) for the HUD.
  Future<void> load({void Function(int stage, double progress)? onStage}) async {
    _seg = await _createTagged(segName, 0, onStage);
    _enc = await _createTagged(encName, 1, onStage);
    _dec = await _createTagged(decName, 2, onStage);
    _warmUp();
  }

  /// Creates one model, tagging any failure with its name so the loading-screen
  /// error names WHICH model failed (e.g. a 404 from an unregistered name).
  Future<ZeticMLangeModel> _createTagged(
    String name,
    int stage,
    void Function(int stage, double progress)? onStage,
  ) async {
    try {
      return await ZeticMLangeModel.create(
        personalKey: personalKey,
        name: name,
        version: 1,
        modelMode: ModelMode.runAuto,
        onProgress: (double p) => onStage?.call(stage, p),
      );
    } catch (e) {
      throw StateError("model '$name' (v1) failed to load: $e");
    }
  }

  void _warmUp() {
    // Segmentation: one 10 s window of silence.
    segmentation(Float32List(160000));
    // Encoder: a zero log-mel -> encoder hidden states.
    final Float32List hidden = encode(Float32List(80 * 3000));
    // Decoder: one step with SOT-seeded buffers + the warm hidden states.
    final Int32List ids = Int32List(448)..[0] = kSotWarm;
    final Int32List mask = Int32List(448)..[0] = 1;
    makeDecoderStep(hidden)(ids, mask);
  }

  static const int kSotWarm = 50258;

  /// Runs segmentation on a `[1,1,160000]` window. Returns flattened `[589*7]`.
  Float32List segmentation(Float32List window160k) {
    final Tensor input =
        Tensor.float32List(window160k, shape: segInShape);
    final List<Tensor> out = _seg!.run(<Tensor>[input]);
    return Float32List.fromList(out.first.asFloat32List());
  }

  /// Runs the encoder on a `[1,80,3000]` log-mel. Returns flattened
  /// `[1500*384]` hidden states (fed to the decoder unchanged).
  Float32List encode(Float32List logMel) {
    final Tensor input = Tensor.float32List(logMel, shape: encInShape);
    final List<Tensor> out = _enc!.run(<Tensor>[input]);
    return Float32List.fromList(out.first.asFloat32List());
  }

  /// Builds a [DecoderStep] bound to a single span's encoder hidden states.
  ///
  /// The hidden-states Tensor is built ONCE and reused across all 448 decode
  /// steps (the SDK copies inputs into its own buffer on each run, so reuse is
  /// safe and avoids re-wrapping 576k floats per step — a Tier B lever).
  /// Positional input order is (ids, enc_hidden, enc_mask); enc_mask is the
  /// 448-long int32 decoder attention mask (GATE-2 decision 8).
  DecoderStep makeDecoderStep(Float32List encHidden) {
    final Tensor encTensor =
        Tensor.float32List(encHidden, shape: encOutShape);
    return (Int32List ids, Int32List mask) {
      final List<Tensor> out = _dec!.run(<Tensor>[
        Tensor.int32List(ids, shape: decTokShape),
        encTensor,
        Tensor.int32List(mask, shape: decTokShape),
      ]);
      // Return the raw output WITHOUT copying. The decoder logits are
      // [1,448,51865] ≈ 93 MB; copying that per greedy step accumulates and
      // triggers an iOS jetsam OOM kill (signal 9). greedyDecode reads only one
      // row and consumes it before the next run(), so the view is valid.
      return out.first.asFloat32List();
    };
  }

  void close() {
    _seg?.close();
    _enc?.close();
    _dec?.close();
    _seg = null;
    _enc = null;
    _dec = null;
  }
}
