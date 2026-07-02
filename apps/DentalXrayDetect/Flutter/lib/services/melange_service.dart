import 'dart:async';
import 'dart:isolate';

import 'package:flutter/foundation.dart';

import '../models/detection.dart';
import 'inference_isolate.dart';
import 'postprocessor.dart';

/// Latency breakdown for the most recent inference (milliseconds).
class FrameTimings {
  const FrameTimings({
    required this.preprocessMs,
    required this.runMs,
    required this.postprocessMs,
  });

  final double preprocessMs;
  final double runMs;
  final double postprocessMs;

  double get totalMs => preprocessMs + runMs + postprocessMs;
}

/// Facade over the dedicated inference isolate.
///
/// Owns the isolate lifecycle, forwards download progress, and enforces a
/// single in-flight request. This is a still-image app (one inference per
/// picked/sample radiograph), so [busy] simply guards against a double-tap
/// while a result is pending.
class MelangeService {
  SendPort? _toIsolate;
  ReceivePort? _fromIsolate;

  Completer<void>? _readyCompleter;
  Completer<InferenceResult>? _pending;
  Completer<SendPort>? _portReady;
  int _requestId = 0;

  bool _busy = false;
  bool _ready = false;

  /// Download progress 0..1 during [init], surfaced for the loading screen.
  final ValueNotifier<double> progress = ValueNotifier<double>(0);

  bool get busy => _busy;
  bool get isReady => _ready;

  /// Spawn the isolate, create + warm up the model. Completes when READY, or
  /// throws with a clear message (e.g. missing key, model not yet READY).
  Future<void> init({
    required String personalKey,
    required String modelName,
    int? version,
    double confThreshold = kDefaultConfThreshold,
    double iouThreshold = kDefaultIouThreshold,
  }) async {
    if (personalKey.isEmpty || personalKey == '<ZETIC_PERSONAL_KEY>') {
      throw StateError(
        'ZETIC personal key is not set. Paste it into '
        'lib/config/secrets.dart (see secrets.example.dart).',
      );
    }

    final ReceivePort fromIsolate = ReceivePort();
    _fromIsolate = fromIsolate;
    _readyCompleter = Completer<void>();
    final Completer<SendPort> portReady = Completer<SendPort>();
    _portReady = portReady;

    await Isolate.spawn(inferenceIsolateEntry, fromIsolate.sendPort);
    fromIsolate.listen(_handleIsolateMessage);

    final SendPort port = await portReady.future;
    _toIsolate = port;
    port.send(InitRequest(
      personalKey: personalKey,
      modelName: modelName,
      version: version,
      confThreshold: confThreshold,
      iouThreshold: iouThreshold,
    ));

    return _readyCompleter!.future;
  }

  void _handleIsolateMessage(dynamic message) {
    if (message is SendPort) {
      _portReady?.complete(message);
      return;
    }
    if (message is ProgressMessage) {
      progress.value = message.progress;
      return;
    }
    if (message is ReadyMessage) {
      _ready = true;
      if (!(_readyCompleter?.isCompleted ?? true)) {
        _readyCompleter!.complete();
      }
      return;
    }
    if (message is ErrorMessage) {
      if (!_ready && !(_readyCompleter?.isCompleted ?? true)) {
        _readyCompleter!.completeError(StateError(message.message));
      }
      if (_pending != null && !_pending!.isCompleted) {
        _pending!.completeError(StateError(message.message));
      }
      return;
    }
    if (message is InferenceResult) {
      final Completer<InferenceResult>? pending = _pending;
      _pending = null;
      _busy = false;
      if (pending != null && !pending.isCompleted) {
        pending.complete(message);
      }
      return;
    }
  }

  /// Submit one still radiograph (packed RGB, [width]x[height]) for detection.
  ///
  /// Returns null if the model isn't ready or a request is already in flight;
  /// otherwise resolves with detections in original-image pixel space
  /// (0..width, 0..height) plus per-stage timings.
  Future<({List<Detection> detections, FrameTimings timings})?> inferStill(
    Uint8List rgb,
    int width,
    int height,
  ) async {
    if (!_ready || _busy || _toIsolate == null) return null;
    _busy = true;
    _requestId++;

    final FrameRequest request = FrameRequest(
      requestId: _requestId,
      width: width,
      height: height,
      rgb: TransferableTypedData.fromList(<Uint8List>[rgb]),
    );
    final Completer<InferenceResult> completer = Completer<InferenceResult>();
    _pending = completer;
    _toIsolate!.send(request);

    try {
      final InferenceResult result = await completer.future;
      return (
        detections: result.detections,
        timings: FrameTimings(
          preprocessMs: result.preprocessMs,
          runMs: result.runMs,
          postprocessMs: result.postprocessMs,
        ),
      );
    } catch (_) {
      return null;
    }
  }

  void dispose() {
    _toIsolate?.send(const DisposeRequest());
    _fromIsolate?.close();
    _toIsolate = null;
    _ready = false;
    progress.dispose();
  }
}
