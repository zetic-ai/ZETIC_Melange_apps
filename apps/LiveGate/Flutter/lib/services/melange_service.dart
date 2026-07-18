import 'package:flutter/foundation.dart';
import 'package:zetic_mlange/zetic_mlange.dart';

import '../config/secrets.dart';
import '../models/gate_result.dart';
import '../models/geometry.dart';
import 'face_align.dart';
import 'frame.dart';
import 'gate.dart';
import 'model_registry.dart';
import 'preprocessor.dart';

/// Request shipped to the [buildInputs] isolate: the raw frame plus the ML Kit
/// face box and landmarks (both in the upright space the frame will be rotated
/// into).
class PrepRequest {
  const PrepRequest({
    required this.frame,
    required this.box,
    required this.landmarks,
  });

  final FrameData frame;
  final FaceBox box;
  final Landmarks5 landmarks;
}

/// Result of the pure-Dart preprocessing pass: both model input tensors (small,
/// cheap to send back across the isolate boundary) plus the measured upright
/// buffer dimensions for the HUD.
class PrepResult {
  const PrepResult({
    required this.padInput,
    required this.faceInput,
    required this.bufferWidth,
    required this.bufferHeight,
  });

  final Float32List padInput;
  final Float32List faceInput;
  final int bufferWidth;
  final int bufferHeight;
}

/// Top-level so it can run inside a [compute] isolate. Converts the frame to
/// upright BGR once, then builds BOTH model inputs from it. Building the face
/// tensor speculatively is cheap (~12k samples); the expensive part — the 38 MB
/// SFace inference — is still skipped on a spoof by the caller.
PrepResult buildInputs(PrepRequest req) {
  final BgrImage img = frameToUprightBgr(req.frame);
  final CropBox crop = computeCropBox(img.width, img.height, req.box);
  final Float32List padInput = buildPadInput(img, crop);
  final Float32List faceInput = buildFaceInput(img, req.landmarks);
  return PrepResult(
    padInput: padInput,
    faceInput: faceInput,
    bufferWidth: img.width,
    bufferHeight: img.height,
  );
}

/// One frame's pipeline output: the gate verdict plus per-stage latencies and
/// the measured buffer size, all surfaced on the HUD (Dart print does not reach
/// the release device console — CLAUDE.md §5).
class PipelineResult {
  const PipelineResult({
    required this.verdict,
    required this.prepMs,
    required this.padMs,
    required this.faceMs,
    required this.bufferWidth,
    required this.bufferHeight,
  });

  final GateVerdict verdict;
  final int prepMs;
  final int padMs;
  final int faceMs;
  final int bufferWidth;
  final int bufferHeight;

  int get totalMs => prepMs + padMs + faceMs;
}

/// Owns the two ZETIC Melange models (PAD + FACE) and the preprocess -> run ->
/// gate pipeline. Both `model.run` calls stay on the calling (main) isolate
/// because the SDK binds each native handle to the isolate that created it; the
/// CPU-heavy Dart preprocessing runs off-thread via [compute].
class MelangeService {
  MelangeService();

  ZeticMLangeModel? _pad;
  ZeticMLangeModel? _face;

  bool get isReady =>
      _pad != null &&
      !_pad!.isClosed &&
      _face != null &&
      !_face!.isClosed;

  /// Downloads (if needed) and initializes BOTH models, then warms each with a
  /// dummy inference so the first real frame is not the slow one. [onProgress]
  /// reports combined 0..1 progress across the two downloads.
  Future<void> init({void Function(double progress)? onProgress}) async {
    if (isReady) return;

    // PAD first (tiny, ~1.8 MB), then FACE (~38 MB) — the bulk of the download.
    _pad = await ZeticMLangeModel.create(
      personalKey: zeticPersonalKey,
      name: ModelRegistry.pad.name,
      version: ModelRegistry.pad.version,
      modelMode: ModelMode.runAuto,
      onProgress: (p) => onProgress?.call(p * 0.15),
    );
    _face = await ZeticMLangeModel.create(
      personalKey: zeticPersonalKey,
      name: ModelRegistry.face.name,
      version: ModelRegistry.face.version,
      modelMode: ModelMode.runAuto,
      onProgress: (p) => onProgress?.call(0.15 + p * 0.85),
    );

    _warmUp();
    onProgress?.call(1.0);
  }

  /// One dummy inference per model so the first live frame is not the cold one.
  void _warmUp() {
    final padDummy = Float32List(3 * kPadInputSize * kPadInputSize);
    _pad!.run([
      Tensor.float32List(padDummy,
          shape: const [1, 3, kPadInputSize, kPadInputSize]),
    ]);
    final faceDummy = Float32List(3 * kFaceInputSize * kFaceInputSize);
    _face!.run([
      Tensor.float32List(faceDummy,
          shape: const [1, 3, kFaceInputSize, kFaceInputSize]),
    ]);
  }

  /// Runs the full pipeline for one detected face. Builds both inputs off-thread,
  /// runs PAD, and only runs the (expensive) FACE model when the face is LIVE
  /// and a reference is enrolled — a spoof never triggers the match, and the
  /// score is withheld by the gate.
  Future<PipelineResult> process(
    PrepRequest req, {
    Float32List? enrolledNormalized,
  }) async {
    final pad = _pad;
    final face = _face;
    if (pad == null || face == null || pad.isClosed || face.isClosed) {
      throw StateError('MelangeService.process called before init()');
    }

    final prepWatch = Stopwatch()..start();
    final prep = await compute(buildInputs, req);
    prepWatch.stop();

    final padWatch = Stopwatch()..start();
    final padOut = pad.run([
      Tensor.float32List(prep.padInput,
          shape: const [1, 3, kPadInputSize, kPadInputSize]),
    ]).first.asFloat32List();
    final padLogits = [padOut[0], padOut[1], padOut[2]];
    padWatch.stop();

    final live = livenessScore(padLogits) >= kLiveThreshold;

    var faceMs = 0;
    Float32List? probe;
    if (live && enrolledNormalized != null) {
      final faceWatch = Stopwatch()..start();
      probe = Float32List.fromList(
        face.run([
          Tensor.float32List(prep.faceInput,
              shape: const [1, 3, kFaceInputSize, kFaceInputSize]),
        ]).first.asFloat32List(),
      );
      faceWatch.stop();
      faceMs = faceWatch.elapsedMilliseconds;
    }

    final verdict = composeVerdict(
      padLogits: padLogits,
      probeEmbedding: probe,
      enrolledNormalized: enrolledNormalized,
    );

    return PipelineResult(
      verdict: verdict,
      prepMs: prepWatch.elapsedMilliseconds,
      padMs: padWatch.elapsedMilliseconds,
      faceMs: faceMs,
      bufferWidth: prep.bufferWidth,
      bufferHeight: prep.bufferHeight,
    );
  }

  /// Computes an L2-normalized embedding for enrollment. Runs the FACE model
  /// unconditionally (enrollment is an explicit user action on a real face).
  Future<Float32List> embed(PrepRequest req) async {
    final face = _face;
    if (face == null || face.isClosed) {
      throw StateError('MelangeService.embed called before init()');
    }
    final prep = await compute(buildInputs, req);
    final raw = Float32List.fromList(
      face.run([
        Tensor.float32List(prep.faceInput,
            shape: const [1, 3, kFaceInputSize, kFaceInputSize]),
      ]).first.asFloat32List(),
    );
    return l2normalize(raw);
  }

  void dispose() {
    if (_pad != null && !_pad!.isClosed) _pad!.close();
    if (_face != null && !_face!.isClosed) _face!.close();
    _pad = null;
    _face = null;
  }
}
