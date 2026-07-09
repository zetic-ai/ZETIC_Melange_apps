import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'inference_worker.dart';

/// UI-side handle to the inference worker isolate. Inference is strictly
/// sequential (one outstanding [run] at a time), so a single pending completer
/// is enough to match a result to its request.
class InferenceClient {
  Isolate? _isolate;
  SendPort? _commands;
  ReceivePort? _fromWorker;
  final _ready = Completer<void>();
  Completer<ResultEvent>? _pending;

  /// 0..1 model-download progress during [start].
  void Function(double progress)? onProgress;

  bool get isReady => _ready.isCompleted;

  /// Spawns the worker, loads the model, resolves when the model is Ready.
  /// Throws (completes with error) if model creation fails.
  Future<void> start({
    required String personalKey,
    required String modelName,
    required List<List<Float32List>> inputsByClip,
    required int c,
    required int h,
    required int w,
  }) async {
    final rp = ReceivePort();
    _fromWorker = rp;
    _isolate = await Isolate.spawn(inferenceIsolateMain, rp.sendPort);

    rp.listen((msg) {
      if (msg is SendPort) {
        _commands = msg;
        msg.send(
          InitRequest(
            personalKey: personalKey,
            modelName: modelName,
            inputsByClip: inputsByClip,
            c: c,
            h: h,
            w: w,
          ),
        );
      } else if (msg is ProgressEvent) {
        onProgress?.call(msg.progress);
      } else if (msg is ReadyEvent) {
        if (!_ready.isCompleted) _ready.complete();
      } else if (msg is ResultEvent) {
        _pending?.complete(msg);
        _pending = null;
      } else if (msg is ErrorEvent) {
        if (!_ready.isCompleted) {
          _ready.completeError(msg.message);
        } else if (_pending != null) {
          _pending!.completeError(msg.message);
          _pending = null;
        }
      }
    });

    return _ready.future;
  }

  /// Runs frame [index] of [clip] and returns its logits + inference latency.
  Future<ResultEvent> run(int clip, int index) {
    final cmd = _commands;
    if (cmd == null) throw StateError('InferenceClient not started');
    assert(_pending == null, 'run() called while a previous run is in flight');
    final c = Completer<ResultEvent>();
    _pending = c;
    cmd.send(RunRequest(clip, index));
    return c.future;
  }

  void dispose() {
    _commands?.send(const CloseRequest());
    _fromWorker?.close();
    // The worker exits itself on CloseRequest; kill as a fallback if it never
    // received the message (e.g. died mid-init).
    _isolate?.kill(priority: Isolate.beforeNextEvent);
    _isolate = null;
  }
}
