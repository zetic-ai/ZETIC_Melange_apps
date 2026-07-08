import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:zetic_mlange/zetic_mlange.dart';

// ---- Messages (client -> worker) ----

class InitRequest {
  InitRequest({
    required this.personalKey,
    required this.modelName,
    required this.inputsByClip,
    required this.c,
    required this.h,
    required this.w,
  });
  final String personalKey;
  final String modelName;
  // [clip][frame] -> flat NCHW input tensor. All clips share the input shape.
  final List<List<Float32List>> inputsByClip;
  final int c, h, w;
}

class RunRequest {
  RunRequest(this.clip, this.index);
  final int clip;
  final int index;
}

class CloseRequest {
  const CloseRequest();
}

// ---- Events (worker -> client) ----

class ProgressEvent {
  ProgressEvent(this.progress);
  final double progress;
}

class ReadyEvent {
  const ReadyEvent();
}

class ResultEvent {
  ResultEvent(this.index, this.logits, this.inferenceMs);
  final int index;
  final Float32List logits; // [1,1,H,W] flattened
  final double inferenceMs;
}

class ErrorEvent {
  ErrorEvent(this.message);
  final String message;
}

/// Worker isolate: owns the ZeticMLangeModel (whose `run()` is synchronous and
/// blocking) so per-frame inference never stalls the UI isolate.
void inferenceIsolateMain(SendPort toClient) {
  final commands = ReceivePort();
  toClient.send(commands.sendPort);

  ZeticMLangeModel? model;
  // Input tensors for every clip are pre-built once and reused every run, so the
  // per-frame loop allocates nothing on the worker heap. [clip][frame] -> Tensor.
  List<List<Tensor>> tensorsByClip = const [];

  commands.listen((msg) async {
    if (msg is InitRequest) {
      tensorsByClip = msg.inputsByClip
          .map(
            (clip) => clip
                .map(
                  (f) => Tensor.float32List(f, shape: [1, msg.c, msg.h, msg.w]),
                )
                .toList(),
          )
          .toList();
      // Melange backend resolution can transiently fail on first launch for some
      // chips (e.g. "failed to resolve backend candidate for rank=1"). Retry a few
      // times before surfacing an error.
      Object? lastErr;
      for (var attempt = 1; attempt <= 4; attempt++) {
        try {
          model = await ZeticMLangeModel.create(
            personalKey: msg.personalKey,
            name: msg.modelName,
            onProgress: (p) => toClient.send(ProgressEvent(p)),
          );
          lastErr = null;
          break;
        } catch (e) {
          lastErr = e;
          model = null;
          // ignore: avoid_print
          print('[medseg] create attempt $attempt failed, retrying: $e');
          toClient.send(
            ProgressEvent(0),
          ); // reset the loading bar for the retry
          await Future<void>.delayed(Duration(milliseconds: 700 * attempt));
        }
      }
      if (model != null) {
        toClient.send(const ReadyEvent());
      } else {
        toClient.send(
          ErrorEvent('model create failed after 4 attempts: $lastErr'),
        );
      }
    } else if (msg is RunRequest) {
      final m = model;
      if (m == null || m.isClosed) return;
      try {
        final sw = Stopwatch()..start();
        final outs = m.run([tensorsByClip[msg.clip][msg.index]]);
        sw.stop();
        // Output tensor views reuse native buffers across runs — copy out now.
        final logits = Float32List.fromList(outs[0].asFloat32List());
        toClient.send(
          ResultEvent(msg.index, logits, sw.elapsedMicroseconds / 1000.0),
        );
      } catch (e) {
        toClient.send(ErrorEvent('run failed: $e'));
      }
    } else if (msg is CloseRequest) {
      model?.close();
      commands.close();
      Isolate.exit();
    }
  });
}
