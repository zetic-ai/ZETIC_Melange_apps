import 'dart:async';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import '../models/speaker_segment.dart';
import '../models/stage_timings.dart';
import '../models/transcript_line.dart';
import 'detokenizer.dart';
import 'diarization_fusion.dart';
import 'log_mel.dart';
import 'melange_service.dart';
import 'postprocessor.dart';
import 'preprocessor.dart';

// ---------------------------------------------------------------------------
// Isolate messages (plain Dart objects; copied across the boundary).
// ---------------------------------------------------------------------------

class _InitMsg {
  _InitMsg(this.reply, this.personalKey, this.melFilters, this.vocabJson);
  final SendPort reply;
  final String personalKey;
  final Float32List melFilters;
  final String vocabJson;
}

class _RunMsg {
  _RunMsg(this.wavBytes);
  final Uint8List wavBytes;
}

class _CloseMsg {
  const _CloseMsg();
}

class _LoadProgressMsg {
  _LoadProgressMsg(this.stage, this.progress);
  final int stage; // 0=segmentation, 1=encoder, 2=decoder
  final double progress;
}

class _ReadyMsg {
  const _ReadyMsg();
}

class _SegmentsMsg {
  _SegmentsMsg(this.segments, this.audioDurationSec);
  final List<SpeakerSegment> segments;
  final double audioDurationSec;
}

class _LineMsg {
  _LineMsg(this.line);
  final TranscriptLine line;
}

class _DoneMsg {
  _DoneMsg(this.timings, this.lines);
  final StageTimings timings;
  final List<TranscriptLine> lines;
}

class _ErrorMsg {
  _ErrorMsg(this.message);
  final String message;
}

class _StatusMsg {
  _StatusMsg(this.text);
  final String text;
}

/// UI-side controller. Spawns and owns the long-lived worker isolate that holds
/// all three model handles and runs the entire pipeline (segmentation + the N×
/// encoder/decoder loops). Heavy tensors never cross the isolate boundary; only
/// short segments / transcript lines / timings do.
class PipelineController {
  PipelineController({
    required this.personalKey,
    required this.melFilters,
    required this.vocabJson,
  });

  final String personalKey;
  final Float32List melFilters;
  final String vocabJson;

  Isolate? _isolate;
  SendPort? _toIsolate;
  ReceivePort? _fromIsolate;

  final Completer<void> _ready = Completer<void>();

  final StreamController<_LoadProgressMsg> _progress =
      StreamController<_LoadProgressMsg>.broadcast();
  final StreamController<List<SpeakerSegment>> _segments =
      StreamController<List<SpeakerSegment>>.broadcast();
  final StreamController<TranscriptLine> _lines =
      StreamController<TranscriptLine>.broadcast();
  final StreamController<StageTimings> _done =
      StreamController<StageTimings>.broadcast();
  final StreamController<String> _status =
      StreamController<String>.broadcast();

  /// (stage 0..2, progress 0..1) during the 3-model cold-start download/warm.
  Stream<({int stage, double progress})> get loadProgress =>
      _progress.stream.map((_LoadProgressMsg m) =>
          (stage: m.stage, progress: m.progress));
  Stream<List<SpeakerSegment>> get segments => _segments.stream;
  Stream<TranscriptLine> get lines => _lines.stream;
  Stream<StageTimings> get done => _done.stream;
  /// Human-readable pipeline breadcrumbs + surfaced error text for the HUD.
  Stream<String> get status => _status.stream;
  Future<void> get ready => _ready.future;

  double _audioDurationSec = 0;
  double get audioDurationSec => _audioDurationSec;

  Future<void> start() async {
    final ReceivePort fromIsolate = ReceivePort();
    _fromIsolate = fromIsolate;
    _isolate = await Isolate.spawn(_entry, fromIsolate.sendPort);

    fromIsolate.listen((Object? message) {
      if (message is SendPort) {
        _toIsolate = message;
        message.send(_InitMsg(
          fromIsolate.sendPort,
          personalKey,
          melFilters,
          vocabJson,
        ));
      } else if (message is _LoadProgressMsg) {
        if (!_progress.isClosed) _progress.add(message);
      } else if (message is _ReadyMsg) {
        if (!_ready.isCompleted) _ready.complete();
      } else if (message is _SegmentsMsg) {
        _audioDurationSec = message.audioDurationSec;
        if (!_segments.isClosed) _segments.add(message.segments);
      } else if (message is _LineMsg) {
        if (!_lines.isClosed) _lines.add(message.line);
      } else if (message is _DoneMsg) {
        if (!_done.isClosed) _done.add(message.timings);
      } else if (message is _StatusMsg) {
        if (!_status.isClosed) _status.add(message.text);
      } else if (message is _ErrorMsg) {
        final StateError err = StateError(message.message);
        // Surface on the status stream so the UI can DISPLAY it (release
        // builds show no Dart logs — CLAUDE.md §5).
        if (!_status.isClosed) _status.add('ERROR: ${message.message}');
        if (!_ready.isCompleted) _ready.completeError(err);
        // Swallow on segments (it has a UI onError too) to avoid an unhandled
        // async error; the status stream is the user-visible channel.
        if (!_segments.isClosed && _segments.hasListener) {
          _segments.addError(err);
        }
      }
    });

    return _ready.future;
  }

  /// Runs the full pipeline on [wavBytes]. Results arrive on [segments],
  /// [lines] (progressively) and [done].
  void runDemo(Uint8List wavBytes) {
    _toIsolate?.send(_RunMsg(wavBytes));
  }

  Future<void> dispose() async {
    _toIsolate?.send(const _CloseMsg());
    _isolate?.kill(priority: Isolate.beforeNextEvent);
    _isolate = null;
    _fromIsolate?.close();
    await _progress.close();
    await _segments.close();
    await _lines.close();
    await _done.close();
    await _status.close();
  }
}

// ---------------------------------------------------------------------------
// Worker isolate entrypoint. Owns the model handles for its whole lifetime.
// ---------------------------------------------------------------------------

Future<void> _entry(SendPort toMain) async {
  final ReceivePort inbox = ReceivePort();
  toMain.send(inbox.sendPort);

  MelangeService? service;
  LogMel? logMel;
  Detokenizer? detok;

  await for (final Object? message in inbox) {
    if (message is _InitMsg) {
      try {
        logMel = LogMel(message.melFilters);
        detok = Detokenizer.fromVocabJson(message.vocabJson);
        service = MelangeService(personalKey: message.personalKey);
        await service.load(
          onStage: (int stage, double p) =>
              message.reply.send(_LoadProgressMsg(stage, p)),
        );
        message.reply.send(const _ReadyMsg());
      } catch (e) {
        message.reply.send(_ErrorMsg('Model load failed: $e'));
      }
    } else if (message is _RunMsg) {
      final MelangeService? svc = service;
      final LogMel? lm = logMel;
      final Detokenizer? dt = detok;
      if (svc == null || lm == null || dt == null || !svc.isLoaded) {
        toMain.send(_ErrorMsg('Run ignored: models not ready'));
        continue;
      }
      try {
        _runPipeline(svc, lm, dt, message.wavBytes, toMain);
      } catch (e, st) {
        toMain.send(_ErrorMsg('Pipeline failed: $e\n$st'));
      }
    } else if (message is _CloseMsg) {
      service?.close();
      break;
    }
  }
  inbox.close();
}

void _runPipeline(
  MelangeService svc,
  LogMel logMel,
  Detokenizer detok,
  Uint8List wavBytes,
  SendPort toMain,
) {
  final StageTimings t = StageTimings();
  final Stopwatch total = Stopwatch()..start();
  final Stopwatch sw = Stopwatch();

  // 1) decode -> mono -> 16 kHz.
  sw.start();
  final Float32List mono = preprocessToMono16k(wavBytes);
  t.decodeWavMs = sw.elapsedMicroseconds / 1000.0;
  t.audioDurationSec = mono.length / kTargetSampleRate;
  toMain.send(_StatusMsg(
      'Decoded ${t.audioDurationSec.toStringAsFixed(1)}s (${mono.length} samples)'));

  // 2) segmentation window -> run.
  sw
    ..reset()
    ..start();
  final Float32List window = segmentationWindow(mono);
  t.segPreMs = sw.elapsedMicroseconds / 1000.0;
  toMain.send(_StatusMsg('Segmenting (CPU)…'));
  sw
    ..reset()
    ..start();
  final Float32List segLogits = svc.segmentation(window);
  t.segRunMs = sw.elapsedMicroseconds / 1000.0;
  toMain.send(_StatusMsg(
      'Segmentation ran in ${t.segRunMs.toStringAsFixed(0)} ms'));

  // Diagnostic: input level + raw seg output stats + argmax class histogram.
  // Offline reference on this clip: in rms~0.142 mx~0.89 | out[-13.4..0.0]
  // cls=116/0/243/208/0/0/22. A mismatch localizes 0-segments (input-silence
  // vs served-artifact). Shown on the HUD (release shows no Dart logs).
  {
    double wSum = 0, wMax = 0;
    for (int i = 0; i < window.length; i++) {
      final double v = window[i];
      wSum += v * v;
      final double a = v < 0 ? -v : v;
      if (a > wMax) wMax = a;
    }
    final double wRms = math.sqrt(wSum / window.length);
    final int nF = segLogits.length ~/ kSegClasses;
    final List<int> hist = List<int>.filled(kSegClasses, 0);
    double oMin = segLogits.isEmpty ? 0 : segLogits[0];
    double oMax = oMin;
    for (int f = 0; f < nF; f++) {
      int best = 0;
      double bv = segLogits[f * kSegClasses];
      for (int c = 0; c < kSegClasses; c++) {
        final double v = segLogits[f * kSegClasses + c];
        if (v > bv) {
          bv = v;
          best = c;
        }
        if (v < oMin) oMin = v;
        if (v > oMax) oMax = v;
      }
      hist[best]++;
    }
    t.diag = 'in rms=${wRms.toStringAsFixed(3)} mx=${wMax.toStringAsFixed(2)} '
        'n=${segLogits.length} | out[${oMin.toStringAsFixed(1)}..'
        '${oMax.toStringAsFixed(1)}] cls=${hist.join("/")}';
    toMain.send(_StatusMsg(t.diag));
  }

  // 3) powerset decode + onset/offset segmentation.
  sw
    ..reset()
    ..start();
  final List<List<bool>> labels = powersetDecode(segLogits);
  final List<SpeakerSegment> segments = onsetOffsetSegments(labels);
  t.powersetMs = sw.elapsedMicroseconds / 1000.0;
  t.segmentsFound = segments.length;
  toMain.send(_StatusMsg('Segments found: ${segments.length}'));

  // Fallback: if diarization yields no segments (e.g. the served segmentation
  // artifact is degenerate), still transcribe the WHOLE clip as one span so the
  // on-device Whisper ASR is visible/proven instead of an empty screen.
  final List<SpeakerSegment> toTranscribe;
  if (segments.isEmpty) {
    toMain.send(_StatusMsg(
        '0 segments — full-clip ASR fallback (diarization artifact)'));
    toTranscribe = <SpeakerSegment>[
      SpeakerSegment(start: 0, end: t.audioDurationSec, speaker: 0),
    ];
  } else {
    toTranscribe = segments;
    toMain.send(_StatusMsg('Transcribing ${segments.length} span(s)…'));
  }
  toMain.send(_SegmentsMsg(toTranscribe, t.audioDurationSec));

  // 4) fusion: per span, encoder + greedy decode + detok (diarize-then-transcribe).
  int spanI = 0;
  final List<TranscriptLine> lines = fuse(
    toTranscribe,
    (SpeakerSegment seg) {
      toMain.send(_StatusMsg(
          'Transcribing span ${++spanI}/${toTranscribe.length} '
          '(spk ${seg.speaker + 1}, ${(seg.end - seg.start).toStringAsFixed(1)}s)…'));
      final int startS = (seg.start * kTargetSampleRate).floor();
      final int endS = (seg.end * kTargetSampleRate).ceil();
      final Float32List span = whisperSpan(mono, startS, endS);

      sw
        ..reset()
        ..start();
      final LogMelResult mel = logMel.compute(span);
      t.logMelMs += sw.elapsedMicroseconds / 1000.0;

      sw
        ..reset()
        ..start();
      final Float32List hidden = svc.encode(mel.data);
      t.encRunMs += sw.elapsedMicroseconds / 1000.0;

      final DecoderStep stepFn = svc.makeDecoderStep(hidden);
      sw
        ..reset()
        ..start();
      final List<int> ids = greedyDecode(stepFn);
      t.decRunMs += sw.elapsedMicroseconds / 1000.0;

      sw
        ..reset()
        ..start();
      final String text = detok.decode(ids);
      t.detokMs += sw.elapsedMicroseconds / 1000.0;
      return text;
    },
    onLine: (TranscriptLine line) => toMain.send(_LineMsg(line)),
  );

  total.stop();
  t.totalMs = total.elapsedMicroseconds / 1000.0;
  toMain.send(_DoneMsg(t, lines));
}
