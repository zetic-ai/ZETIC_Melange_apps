import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';
import 'dart:ui';

import 'db_postprocessor.dart';
import 'detector_preprocessor.dart';
import 'frame_data.dart';
import 'recognizer_preprocessor.dart';

/// Detector-preprocess stage result returned to the main isolate.
class DetFrameResult {
  const DetFrameResult({
    required this.input,
    required this.geometry,
    required this.bufferWidth,
    required this.bufferHeight,
  });

  /// Flattened [1,3,640,640] detector input.
  final Float32List input;
  final LetterboxGeometry geometry;

  /// Raw camera buffer dimensions as delivered (pre-rotation) — surfaced on
  /// the HUD so the real orientation can be verified on-device.
  final int bufferWidth;
  final int bufferHeight;
}

/// A single long-lived worker isolate hosting the CPU-heavy pure-Dart
/// pipeline stages, replacing PyroGuard's per-frame `compute()` double-spawn
/// (~20 ms/frame measured there).
///
/// Per frame, three awaited round-trips (one request in flight at a time):
///  1. [prepareFrame] — raw camera planes in; the frame stays RESIDENT in the
///     worker; letterboxed/normalized detector tensor out.
///  2. [decodeHeatmap] — detector heatmap in; DB decode against the stored
///     letterbox geometry; text regions + heatmap stats out.
///  3. [cropForRecognition] — budgeted quads in; each cropped from the
///     resident full-resolution frame, warped upright, padded/normalized;
///     recognizer tensors out.
///
/// `model.run` itself stays on the main isolate (the Melange handle is bound
/// to it). CTC decode and PII classification are trivial and also run on main.
class DocPipeline {
  DocPipeline._(this._isolate, this._toWorker, this._fromWorker);

  final Isolate _isolate;
  final SendPort _toWorker;
  final ReceivePort _fromWorker;

  Completer<Object?>? _pending;
  StreamSubscription<Object?>? _sub;

  static Future<DocPipeline> spawn() async {
    final fromWorker = ReceivePort();
    // Single broadcast stream: first event is the worker's SendPort handshake,
    // everything after is request responses. The worker only ever sends a
    // response after receiving a request, so nothing can be dropped between
    // the handshake and attaching the response listener below.
    final stream = fromWorker.asBroadcastStream();
    final isolate = await Isolate.spawn(_workerMain, fromWorker.sendPort);
    final toWorker = await stream.first as SendPort;

    final pipeline = DocPipeline._(isolate, toWorker, fromWorker);
    pipeline._sub = stream.listen(pipeline._onMessage);
    return pipeline;
  }

  void _onMessage(Object? message) {
    final pending = _pending;
    _pending = null;
    pending?.complete(message);
  }

  Future<T> _request<T>(Object message) {
    assert(_pending == null, 'DocPipeline allows one request in flight');
    final completer = Completer<Object?>();
    _pending = completer;
    _toWorker.send(message);
    return completer.future.then((v) {
      if (v is _WorkerError) {
        throw StateError('Pipeline isolate error: ${v.message}');
      }
      return v as T;
    });
  }

  Future<DetFrameResult> prepareFrame(FrameData frame) =>
      _request<DetFrameResult>(_PrepareFrameMsg(frame));

  Future<DbDecodeResult> decodeHeatmap(Float32List heatmap) =>
      _request<DbDecodeResult>(_DecodeHeatmapMsg(heatmap));

  Future<List<Float32List>> cropForRecognition(List<List<Offset>> quads) =>
      _request<List<Float32List>>(_CropMsg(quads));

  void dispose() {
    _sub?.cancel();
    _fromWorker.close();
    _isolate.kill(priority: Isolate.immediate);
  }
}

class _PrepareFrameMsg {
  const _PrepareFrameMsg(this.frame);
  final FrameData frame;
}

class _DecodeHeatmapMsg {
  const _DecodeHeatmapMsg(this.heatmap);
  final Float32List heatmap;
}

class _CropMsg {
  const _CropMsg(this.quads);
  final List<List<Offset>> quads;
}

class _WorkerError {
  const _WorkerError(this.message);
  final String message;
}

void _workerMain(SendPort toMain) {
  final fromMain = ReceivePort();
  toMain.send(fromMain.sendPort);

  // Worker state: the resident frame + geometry, and reused buffers.
  UprightFrame? frame;
  LetterboxGeometry? geometry;
  final detBuffer = Float32List(3 * kDetInputSize * kDetInputSize);

  fromMain.listen((Object? message) {
    try {
      if (message is _PrepareFrameMsg) {
        frame = UprightFrame(message.frame);
        final result = preprocessDetectorFrame(frame!, out: detBuffer);
        geometry = result.geometry;
        // The Float32List is copied by send(); detBuffer stays reusable here.
        toMain.send(DetFrameResult(
          input: result.input,
          geometry: result.geometry,
          bufferWidth: message.frame.width,
          bufferHeight: message.frame.height,
        ));
      } else if (message is _DecodeHeatmapMsg) {
        final geom = geometry;
        if (geom == null) {
          toMain.send(const _WorkerError('decodeHeatmap before prepareFrame'));
          return;
        }
        toMain.send(decodeDbHeatmap(message.heatmap, geom));
      } else if (message is _CropMsg) {
        final src = frame;
        if (src == null) {
          toMain.send(
              const _WorkerError('cropForRecognition before prepareFrame'));
          return;
        }
        final tensors = message.quads
            .map((q) => preprocessRecognizerCrop(src, q))
            .toList(growable: false);
        toMain.send(tensors);
      } else {
        toMain.send(_WorkerError('unknown message ${message.runtimeType}'));
      }
    } catch (e) {
      toMain.send(_WorkerError('$e'));
    }
  });
}
