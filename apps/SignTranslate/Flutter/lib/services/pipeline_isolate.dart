import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';
import 'dart:ui' show Offset;

import '../config.dart';
import '../models/text_region.dart';
import 'db_postprocessor.dart';
import 'detector_preprocessor.dart';
import 'frame_data.dart';
import 'quad_deskew.dart';
import 'rec_preprocessor.dart';

/// Detection preprocessing result delivered back to the main isolate.
class DetPrepResult {
  const DetPrepResult({required this.input, required this.geometry});

  /// NCHW [1,3,736,736] detector input, ready to wrap as a Tensor.
  final Float32List input;
  final LetterboxGeometry geometry;
}

/// DB postprocessing result: quads in upright FRAME space + heatmap stats.
class DbPostResult {
  const DbPostResult({
    required this.quads,
    required this.mapMin,
    required this.mapMax,
    required this.mapMean,
  });

  final List<Quad> quads;
  final double mapMin;
  final double mapMax;
  final double mapMean;
}

/// ONE long-lived pipeline isolate for all heavy pure-Dart work (BGR
/// conversion, letterbox-736 preprocessing, DB postprocessing, per-crop
/// deskew + recognizer preprocessing).
///
/// Rationale (adopted from the approved LiveDocRedact architecture): a
/// per-frame `compute()` costs ~20 ms in double isolate-spawn overhead
/// (measured on PyroGuard). Here the isolate lives for the whole session,
/// frame bytes cross the boundary once per detection pass, and the isolate
/// RETAINS the upright BGR frame so staggered per-crop recognition on later
/// frames needs no further frame traffic. Both `model.run` calls stay on the
/// main isolate — the Melange model handles are bound to it.
///
/// The isolate owns pre-allocated input tensors ([1,3,736,736] and
/// [1,3,48,320]) and writes them in place every pass.
class PipelineWorker {
  PipelineWorker._(this._isolate, this._toIsolate, this._fromIsolate) {
    _subscription = _fromIsolate.listen(_onReply);
  }

  final Isolate _isolate;
  final SendPort _toIsolate;
  final ReceivePort _fromIsolate;
  late final StreamSubscription<dynamic> _subscription;

  final Map<int, Completer<List<dynamic>>> _pending = {};
  int _nextId = 0;
  bool _disposed = false;

  static Future<PipelineWorker> spawn() async {
    final handshake = ReceivePort();
    final isolate = await Isolate.spawn(
      _pipelineMain,
      handshake.sendPort,
      debugName: 'glyphgo-pipeline',
    );
    final fromIsolate = ReceivePort();
    final toIsolate = await handshake.first as SendPort;
    toIsolate.send(fromIsolate.sendPort);
    handshake.close();
    return PipelineWorker._(isolate, toIsolate, fromIsolate);
  }

  void _onReply(dynamic message) {
    final reply = message as List<dynamic>;
    final id = reply[0] as int;
    final completer = _pending.remove(id);
    if (completer == null) return;
    if (reply[1] == 'err') {
      completer.completeError(StateError(reply[2] as String));
    } else {
      completer.complete(reply.sublist(2));
    }
  }

  Future<List<dynamic>> _request(String op, List<dynamic> args) {
    if (_disposed) {
      return Future.error(StateError('PipelineWorker is disposed'));
    }
    final id = _nextId++;
    final completer = Completer<List<dynamic>>();
    _pending[id] = completer;
    _toIsolate.send([id, op, ...args]);
    return completer.future;
  }

  /// Ships one camera frame to the isolate; returns the detector input and
  /// letterbox geometry. The isolate retains the upright BGR frame for
  /// subsequent [prepareCrop] calls.
  Future<DetPrepResult> prepareDetection(FrameData frame) async {
    final r = await _request('detPrep', [frame]);
    return DetPrepResult(
      input: r[0] as Float32List,
      geometry: LetterboxGeometry(
        scale: r[1] as double,
        padX: r[2] as int,
        padY: r[3] as int,
        srcWidth: r[4] as int,
        srcHeight: r[5] as int,
      ),
    );
  }

  /// Runs DB postprocessing on the detector heatmap against the retained
  /// letterbox geometry; returns frame-space quads in reading order.
  Future<DbPostResult> postprocessDetection(Float32List heatmap) async {
    final r = await _request('dbPost', [heatmap]);
    final flat = r[0] as Float64List;
    final quads = <Quad>[
      for (var i = 0; i + 7 < flat.length; i += 8)
        Quad(
          Offset(flat[i], flat[i + 1]),
          Offset(flat[i + 2], flat[i + 3]),
          Offset(flat[i + 4], flat[i + 5]),
          Offset(flat[i + 6], flat[i + 7]),
        ),
    ];
    return DbPostResult(
      quads: quads,
      mapMin: r[1] as double,
      mapMax: r[2] as double,
      mapMean: r[3] as double,
    );
  }

  /// Deskews one quad (upright frame space) out of the RETAINED frame and
  /// returns the normalized [1,3,48,320] recognizer input.
  Future<Float32List> prepareCrop(Quad quad) async {
    final coords = Float64List(8);
    final pts = quad.points;
    for (var i = 0; i < 4; i++) {
      coords[2 * i] = pts[i].dx;
      coords[2 * i + 1] = pts[i].dy;
    }
    final r = await _request('recPrep', [coords]);
    return r[0] as Float32List;
  }

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    for (final c in _pending.values) {
      c.completeError(StateError('PipelineWorker disposed'));
    }
    _pending.clear();
    _subscription.cancel();
    _fromIsolate.close();
    _isolate.kill(priority: Isolate.immediate);
  }
}

// ---------------------------------------------------------------------------
// Isolate side.
// ---------------------------------------------------------------------------

Future<void> _pipelineMain(SendPort handshake) async {
  final inbox = ReceivePort();
  handshake.send(inbox.sendPort);

  SendPort? outbox;

  // Pre-allocated tensors + retained per-pass state.
  final detInput = Float32List(3 * kDetInputSize * kDetInputSize);
  final recInput = Float32List(3 * kRecHeight * kRecWidth);
  BgrFrame? frame;
  LetterboxGeometry? geometry;

  await for (final message in inbox) {
    if (message is SendPort) {
      outbox = message;
      continue;
    }
    final request = message as List<dynamic>;
    final id = request[0] as int;
    final op = request[1] as String;
    try {
      switch (op) {
        case 'detPrep':
          frame = convertToUprightBgr(request[2] as FrameData);
          geometry = letterboxDetectorInput(frame, detInput);
          outbox!.send([
            id,
            'ok',
            detInput, // copied by send()
            geometry.scale,
            geometry.padX,
            geometry.padY,
            geometry.srcWidth,
            geometry.srcHeight,
          ]);
        case 'dbPost':
          final geo = geometry;
          if (geo == null) throw StateError('dbPost before detPrep');
          final result = dbPostProcess(request[2] as Float32List, geo);
          final flat = Float64List(result.quads.length * 8);
          for (var q = 0; q < result.quads.length; q++) {
            final pts = result.quads[q].points;
            for (var i = 0; i < 4; i++) {
              flat[q * 8 + 2 * i] = pts[i].dx;
              flat[q * 8 + 2 * i + 1] = pts[i].dy;
            }
          }
          outbox!.send([
            id,
            'ok',
            flat,
            result.mapMin,
            result.mapMax,
            result.mapMean,
          ]);
        case 'recPrep':
          final f = frame;
          if (f == null) throw StateError('recPrep before detPrep');
          final coords = request[2] as Float64List;
          final quad = Quad(
            Offset(coords[0], coords[1]),
            Offset(coords[2], coords[3]),
            Offset(coords[4], coords[5]),
            Offset(coords[6], coords[7]),
          );
          final crop = deskewQuad(f, quad);
          recognizerPreprocess(crop, recInput);
          outbox!.send([id, 'ok', recInput]); // copied by send()
        case 'close':
          inbox.close();
        default:
          throw StateError('Unknown pipeline op: $op');
      }
    } catch (e) {
      outbox?.send([id, 'err', '$e']);
    }
  }
}
