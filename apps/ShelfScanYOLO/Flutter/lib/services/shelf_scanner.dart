import 'dart:typed_data';

import '../models/detection.dart';
import 'letterbox.dart';
import 'melange_service.dart';
import 'nms.dart';
import 'postprocessor.dart';
import 'preprocessor.dart';

/// One full scan's output plus per-stage timings for the on-screen HUD.
class ScanResult {
  const ScanResult({
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
    required this.transform,
    required this.rawKept,
    required this.firstRawBox,
    required this.preprocessMs,
    required this.inferenceMs,
    required this.postprocessMs,
  });

  final List<Detection> detections;
  final int imageWidth;
  final int imageHeight;
  final LetterboxTransform transform;

  /// Count that passed the confidence threshold BEFORE NMS (debug signal).
  final int rawKept;
  final BBox? firstRawBox;

  final double preprocessMs;
  final double inferenceMs;
  final double postprocessMs;

  int get count => detections.length;
  double get totalMs => preprocessMs + inferenceMs + postprocessMs;
}

/// Composes preprocess -> Melange inference -> decode -> NMS for a still image,
/// timing each stage. Everything runs on the calling (main) isolate because the
/// Melange handle is isolate-bound; this is a one-shot per user action.
class ShelfScanner {
  ShelfScanner({
    required this.melange,
    this.preprocessor = const Preprocessor(),
    this.postprocessor = const Postprocessor(),
  });

  final MelangeService melange;
  final Preprocessor preprocessor;
  final Postprocessor postprocessor;

  ScanResult scan(Uint8List encodedBytes) {
    final sw = Stopwatch()..start();

    final pre = preprocessor.process(encodedBytes);
    final preMs = sw.elapsedMicroseconds / 1000.0;

    sw
      ..reset()
      ..start();
    final output = melange.runInference(pre.input);
    final infMs = sw.elapsedMicroseconds / 1000.0;

    sw
      ..reset()
      ..start();
    final raw = postprocessor.decode(output, pre.transform);
    final detections = nonMaxSuppression(raw, postprocessor.iouThreshold);
    final postMs = sw.elapsedMicroseconds / 1000.0;

    return ScanResult(
      detections: detections,
      imageWidth: pre.originalWidth,
      imageHeight: pre.originalHeight,
      transform: pre.transform,
      rawKept: raw.length,
      firstRawBox: raw.isNotEmpty ? raw.first.box : null,
      preprocessMs: preMs,
      inferenceMs: infMs,
      postprocessMs: postMs,
    );
  }
}
