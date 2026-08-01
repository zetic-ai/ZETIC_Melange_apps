import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:zetic_mlange/zetic_mlange.dart';

import '../models/detection.dart';
import 'frame_data.dart';
import 'postprocessor.dart';
import 'preprocessor.dart';

/// Result of one inference: detections plus the upright image dimensions the
/// boxes live in (so the overlay can map image-space -> screen) and the
/// measured isolate-side latency.
class InferenceResult {
  const InferenceResult(
    this.detections,
    this.imageWidth,
    this.imageHeight,
    this.latencyMs,
  );

  final List<Detection> detections;
  final int imageWidth;
  final int imageHeight;
  final double latencyMs;

  static const empty = InferenceResult(<Detection>[], 0, 0, 0);
}

/// Owns the **long-lived dedicated inference isolate**.
///
/// The Melange FFI model handle is bound to the isolate that creates it, so the
/// isolate owns the whole lifecycle: create -> warm-up -> run -> close. The main
/// isolate sends frame bytes in (one copy) and gets a tiny result list back.
/// This dedicated-isolate posture (chosen over inline) keeps the UI smooth and
/// robust to a server-side CPU-fallback flip. A `_busy` guard drops frames so
/// they never pile up.
class MelangeService {
  MelangeService({
    required this.modelName,
    required this.modelVersion,
  });

  final String modelName;
  final int modelVersion;

  Isolate? _isolate;
  SendPort? _toIsolate;
  ReceivePort? _fromIsolate;
  StreamSubscription<dynamic>? _sub;

  Completer<void>? _initCompleter;
  Completer<InferenceResult>? _inflight;
  bool _busy = false;
  bool _closed = false;

  bool get isBusy => _busy;

  /// Spawns the isolate, creates the model and runs a warm-up inference.
  /// Throws if [personalKey] is empty or model creation fails.
  Future<void> init({required String personalKey}) async {
    if (personalKey.isEmpty) {
      throw StateError(
        'ZETIC personal key is empty. Build with '
        '--dart-define=ZETIC_KEY=<your_key>.',
      );
    }
    _initCompleter = Completer<void>();
    _fromIsolate = ReceivePort();
    _sub = _fromIsolate!.listen(_onIsolateMessage);

    _isolate = await Isolate.spawn(
      _isolateEntry,
      _IsolateConfig(
        replyTo: _fromIsolate!.sendPort,
        personalKey: personalKey,
        modelName: modelName,
        modelVersion: modelVersion,
      ),
    );
    return _initCompleter!.future;
  }

  /// Runs inference on one frame. Returns null immediately if the isolate is
  /// still busy with the previous frame (frame-drop, not queue).
  Future<InferenceResult?> detect(FrameData frame) {
    if (_closed || _toIsolate == null || _busy) {
      return Future.value(null);
    }
    _busy = true;
    _inflight = Completer<InferenceResult>();
    _toIsolate!.send(_serializeFrame(frame));
    return _inflight!.future;
  }

  void _onIsolateMessage(dynamic msg) {
    if (msg is SendPort) {
      _toIsolate = msg;
      return;
    }
    if (msg is! Map) return;
    switch (msg['type']) {
      case 'ready':
        _initCompleter?.complete();
        break;
      case 'initError':
        _initCompleter?.completeError(StateError(msg['message'] as String));
        break;
      case 'result':
        _busy = false;
        _inflight?.complete(_deserializeResult(msg));
        _inflight = null;
        break;
      case 'runError':
        _busy = false;
        _inflight?.complete(InferenceResult.empty);
        _inflight = null;
        break;
    }
  }

  void close() {
    _closed = true;
    _toIsolate?.send({'type': 'close'});
    _sub?.cancel();
    _fromIsolate?.close();
    _isolate?.kill(priority: Isolate.beforeNextEvent);
    _isolate = null;
    _toIsolate = null;
  }

  // --- serialization across the isolate boundary (sendable primitives only) ---

  static List<dynamic> _serializeFrame(FrameData f) => <dynamic>[
    f.format.index,
    f.width,
    f.height,
    f.rotationDegrees,
    f.plane0,
    f.bytesPerRow0,
    f.plane1,
    f.plane2,
    f.bytesPerRow1,
    f.bytesPerRow2,
    f.pixelStride1,
    f.pixelStride2,
  ];

  static InferenceResult _deserializeResult(Map msg) {
    final boxes = msg['boxes'] as Float32List;
    final imageW = msg['imageW'] as int;
    final imageH = msg['imageH'] as int;
    final latency = msg['latencyMs'] as double;
    final dets = <Detection>[];
    for (var i = 0; i + 4 < boxes.length; i += 5) {
      dets.add(
        Detection(
          left: boxes[i],
          top: boxes[i + 1],
          right: boxes[i + 2],
          bottom: boxes[i + 3],
          confidence: boxes[i + 4],
        ),
      );
    }
    return InferenceResult(dets, imageW, imageH, latency);
  }
}

class _IsolateConfig {
  const _IsolateConfig({
    required this.replyTo,
    required this.personalKey,
    required this.modelName,
    required this.modelVersion,
  });

  final SendPort replyTo;
  final String personalKey;
  final String modelName;
  final int modelVersion;
}

/// Entry point of the dedicated inference isolate. Owns the FFI model handle.
Future<void> _isolateEntry(_IsolateConfig cfg) async {
  final commands = ReceivePort();
  cfg.replyTo.send(commands.sendPort);

  ZeticMLangeModel? model;
  try {
    model = await ZeticMLangeModel.create(
      personalKey: cfg.personalKey,
      name: cfg.modelName,
      version: cfg.modelVersion,
      modelMode: ModelMode.runAuto,
    );
    // Warm-up: first inference compiles/loads the backend; do it now so the
    // first real frame is not the slow one.
    final warm = Float32List(3 * 640 * 640);
    model.run([Tensor.float32List(warm, shape: const [1, 3, 640, 640])]);
    cfg.replyTo.send({'type': 'ready'});
  } catch (e) {
    cfg.replyTo.send({'type': 'initError', 'message': 'Model init failed: $e'});
    commands.close();
    return;
  }

  final preprocessor = Preprocessor();
  const postprocessor = Postprocessor();

  await for (final msg in commands) {
    if (msg is Map && msg['type'] == 'close') {
      break;
    }
    if (msg is! List) continue;
    final sw = Stopwatch()..start();
    try {
      final frame = _deserializeFrame(msg);
      final pre = preprocessor.process(frame);
      final outputs = model.run([
        Tensor.float32View(pre.input, shape: const [1, 3, 640, 640]),
      ]);
      final raw = outputs.first.asFloat32List();
      final dets = postprocessor.decode(raw, pre.params);
      sw.stop();
      cfg.replyTo.send({
        'type': 'result',
        'boxes': _serializeDetections(dets),
        'imageW': pre.params.srcWidth,
        'imageH': pre.params.srcHeight,
        'latencyMs': sw.elapsedMicroseconds / 1000.0,
      });
    } catch (e) {
      cfg.replyTo.send({'type': 'runError', 'message': '$e'});
    }
  }

  model.close();
  commands.close();
}

FrameData _deserializeFrame(List<dynamic> m) => FrameData(
  format: FramePixelFormat.values[m[0] as int],
  width: m[1] as int,
  height: m[2] as int,
  rotationDegrees: m[3] as int,
  plane0: m[4] as Uint8List,
  bytesPerRow0: m[5] as int,
  plane1: m[6] as Uint8List?,
  plane2: m[7] as Uint8List?,
  bytesPerRow1: m[8] as int,
  bytesPerRow2: m[9] as int,
  pixelStride1: m[10] as int,
  pixelStride2: m[11] as int,
);

Float32List _serializeDetections(List<Detection> dets) {
  final out = Float32List(dets.length * 5);
  for (var i = 0; i < dets.length; i++) {
    final d = dets[i];
    out[i * 5] = d.left;
    out[i * 5 + 1] = d.top;
    out[i * 5 + 2] = d.right;
    out[i * 5 + 3] = d.bottom;
    out[i * 5 + 4] = d.confidence;
  }
  return out;
}
