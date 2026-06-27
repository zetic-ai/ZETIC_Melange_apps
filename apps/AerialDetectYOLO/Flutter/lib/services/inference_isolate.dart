import 'dart:isolate';
import 'dart:typed_data';

import 'package:zetic_mlange/zetic_mlange.dart';

import '../models/detection.dart';
import 'postprocessor.dart';
import 'preprocessor.dart';

/// Camera pixel formats the isolate knows how to decode.
enum FrameFormat { bgra8888, yuv420 }

/// A frame handed to the inference isolate. Plane bytes are wrapped in
/// [TransferableTypedData] so they move across the isolate boundary by
/// ownership transfer (no copy) — frame bytes in, [Detection]s out.
class FrameRequest {
  FrameRequest({
    required this.requestId,
    required this.format,
    required this.width,
    required this.height,
    required this.bytesPerRow,
    required this.plane0,
    this.plane1,
    this.plane2,
    this.uvRowStride = 0,
    this.uvPixelStride = 0,
  });

  final int requestId;
  final FrameFormat format;
  final int width;
  final int height;
  final int bytesPerRow;
  final TransferableTypedData plane0;
  final TransferableTypedData? plane1;
  final TransferableTypedData? plane2;
  final int uvRowStride;
  final int uvPixelStride;
}

/// Result for one [FrameRequest]. All fields are primitives / primitive-field
/// objects so it copies cleanly back to the UI isolate.
class InferenceResult {
  InferenceResult({
    required this.requestId,
    required this.detections,
    required this.preprocessMs,
    required this.runMs,
    required this.postprocessMs,
  });

  final int requestId;
  final List<Detection> detections;
  final double preprocessMs;
  final double runMs;
  final double postprocessMs;
}

// ---- Control messages (UI isolate -> worker isolate) ----

class InitRequest {
  InitRequest({
    required this.personalKey,
    required this.modelName,
    required this.version,
    required this.confThreshold,
    required this.iouThreshold,
  });

  final String personalKey;
  final String modelName;
  final int? version;
  final double confThreshold;
  final double iouThreshold;
}

class DisposeRequest {
  const DisposeRequest();
}

// ---- Status messages (worker isolate -> UI isolate) ----

class ProgressMessage {
  ProgressMessage(this.progress);
  final double progress;
}

class ReadyMessage {
  const ReadyMessage();
}

class ErrorMessage {
  ErrorMessage(this.message);
  final String message;
}

/// Entry point of the long-lived inference isolate.
///
/// The Melange model is created, warmed up, and run entirely inside this single
/// isolate (the SDK binds the native model handle to the creating isolate), so
/// the 928x928 preprocess + run + ~247K-float decode never block the UI isolate
/// — a CPU-fallback latency flip janks nothing on screen.
void inferenceIsolateEntry(SendPort toMain) {
  final ReceivePort fromMain = ReceivePort();
  toMain.send(fromMain.sendPort);

  ZeticMLangeModel? model;
  double confThreshold = kDefaultConfThreshold;
  double iouThreshold = kDefaultIouThreshold;

  // Preallocated NCHW input buffer, reused for every frame (Tier B: no per-frame
  // allocation of a ~10MB Float32List).
  final Float32List inputBuffer = Float32List(kInputElements);

  fromMain.listen((dynamic message) async {
    if (message is InitRequest) {
      confThreshold = message.confThreshold;
      iouThreshold = message.iouThreshold;
      try {
        model = await ZeticMLangeModel.create(
          personalKey: message.personalKey,
          name: message.modelName,
          version: message.version,
          modelMode: ModelMode.runAuto,
          onProgress: (double p) => toMain.send(ProgressMessage(p)),
        );
        // Warm-up: one dummy inference so the first real frame isn't the slow
        // one. NOTE: this is a live model.run() — it only executes on-device
        // once the Melange model reaches READY (parked until then).
        final Tensor warm = Tensor.float32List(
          inputBuffer,
          shape: <int>[1, 3, kInputSize, kInputSize],
        );
        model!.run(<Tensor>[warm]);
        toMain.send(const ReadyMessage());
      } catch (e) {
        toMain.send(ErrorMessage('Model init failed: $e'));
      }
      return;
    }

    if (message is FrameRequest) {
      final ZeticMLangeModel? m = model;
      if (m == null) {
        toMain.send(InferenceResult(
          requestId: message.requestId,
          detections: const <Detection>[],
          preprocessMs: 0,
          runMs: 0,
          postprocessMs: 0,
        ));
        return;
      }
      try {
        final LetterboxParams lb =
            computeLetterbox(message.width, message.height);

        // --- preprocess (fused into inputBuffer) ---
        final Stopwatch sw = Stopwatch()..start();
        switch (message.format) {
          case FrameFormat.bgra8888:
            letterboxBgraToNchw(
              message.plane0.materialize().asUint8List(),
              message.width,
              message.height,
              message.bytesPerRow,
              lb,
              inputBuffer,
            );
          case FrameFormat.yuv420:
            final Uint8List rgb = yuv420ToRgb(
              message.plane0.materialize().asUint8List(),
              message.plane1!.materialize().asUint8List(),
              message.plane2!.materialize().asUint8List(),
              message.width,
              message.height,
              message.bytesPerRow,
              message.uvRowStride,
              message.uvPixelStride,
            );
            letterboxRgbToNchw(
              rgb, message.width, message.height, lb, inputBuffer);
        }
        final double preprocessMs = sw.elapsedMicroseconds / 1000.0;

        // --- run (device-blocked until model READY) ---
        sw
          ..reset()
          ..start();
        final Tensor input = Tensor.float32List(
          inputBuffer,
          shape: <int>[1, 3, kInputSize, kInputSize],
        );
        final List<Tensor> outputs = m.run(<Tensor>[input]);
        final Float32List raw = outputs.first.asFloat32List();
        final double runMs = sw.elapsedMicroseconds / 1000.0;

        // --- postprocess (decode + per-class NMS) ---
        sw
          ..reset()
          ..start();
        final List<Detection> dets = decodeDetections(
          raw,
          lb,
          confThreshold: confThreshold,
          iouThreshold: iouThreshold,
        );
        final double postprocessMs = sw.elapsedMicroseconds / 1000.0;

        toMain.send(InferenceResult(
          requestId: message.requestId,
          detections: dets,
          preprocessMs: preprocessMs,
          runMs: runMs,
          postprocessMs: postprocessMs,
        ));
      } catch (e) {
        toMain.send(ErrorMessage('Inference failed: $e'));
        toMain.send(InferenceResult(
          requestId: message.requestId,
          detections: const <Detection>[],
          preprocessMs: 0,
          runMs: 0,
          postprocessMs: 0,
        ));
      }
      return;
    }

    if (message is DisposeRequest) {
      model?.close();
      model = null;
      fromMain.close();
      Isolate.exit();
    }
  });
}
