import 'dart:typed_data';

import 'package:zetic_mlange/zetic_mlange.dart';

import '../config/secrets.dart';
import '../models/phonemes.dart';
import '../models/scoring.dart';
import '../models/sentence.dart';
import 'gop_scorer.dart';
import 'postprocessor.dart';
import 'preprocessor.dart';

/// Owns the ZETIC Melange model lifecycle and runs the full
/// preprocess -> inference -> score pipeline for one recording.
///
/// The model runs on the main isolate: there is exactly ONE inference per 5.11 s
/// recording (not per-frame), and the pure-Dart scoring head is sub-millisecond,
/// so a per-run compute() isolate would be net overhead.
class MelangeService {
  MelangeService({GopScorer? scorer}) : _scorer = scorer ?? const GopScorer();

  static const String modelName = 'ajayshah/PronunciationScoring';
  static const int modelVersion = 1;

  final GopScorer _scorer;
  ZeticMLangeModel? _model;

  /// The backend artifact / mode the last run reported, for the HUD.
  String servedArtifact = 'RUN_AUTO';

  bool get isReady => _model != null && !_model!.isClosed;

  /// Download (first launch) + initialize the model, then warm it up with one
  /// dummy inference so the first real recording isn't penalized by lazy setup.
  Future<void> init({void Function(double progress)? onProgress}) async {
    if (isReady) return;
    _model = await ZeticMLangeModel.create(
      personalKey: zeticPersonalKey,
      name: modelName,
      version: modelVersion,
      modelMode: ModelMode.runAuto,
      onProgress: onProgress,
    );
    _warmUp();
  }

  void _warmUp() {
    final model = _model;
    if (model == null || model.isClosed) return;
    final dummy = Float32List(kTargetSamples); // silence is fine for warm-up
    final input = Tensor.float32List(dummy, shape: const [1, kTargetSamples]);
    model.run([input]);
  }

  /// Score one raw PCM16 capture against [sentence].
  PronunciationResult score(
    Uint8List pcmBytes,
    int rate,
    PracticeSentence sentence,
  ) {
    final model = _model;
    if (model == null || model.isClosed) {
      throw StateError('MelangeService.score called before init()');
    }

    final mode = resolveRate(rate);
    final input = buildModelInput(pcmBytes, rate);

    final inferSw = Stopwatch()..start();
    final tensor =
        Tensor.float32List(input, shape: const [1, kTargetSamples]);
    final outputs = model.run([tensor]);
    // Copy out of the reused native buffer before the next run can clobber it.
    final logits = Float32List.fromList(outputs.first.asFloat32List());
    inferSw.stop();

    final scoreSw = Stopwatch()..start();
    final lp = LogProbView(logits);
    final greedy = greedyDecode(lp);
    final words = _scorer.scoreWords(lp, sentence, greedy.blankFraction);
    final overall = _scorer.overall(words);
    scoreSw.stop();

    return PronunciationResult(
      words: words,
      overallScore: overall,
      blankFraction: greedy.blankFraction,
      greedyPhonemes: greedy.phonemes,
      inferenceMs: inferSw.elapsedMilliseconds,
      scoringMs: scoreSw.elapsedMilliseconds,
      sampleRateInfo: rateModeLabel(mode),
    );
  }

  void dispose() {
    final model = _model;
    if (model != null && !model.isClosed) model.close();
    _model = null;
  }
}

/// Exposed for asserting the model's fixed output geometry in tests.
const int kExpectedFrames = Phonemes.frameCount;
const int kExpectedClasses = Phonemes.classCount;
