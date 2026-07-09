import 'dart:async';
import 'dart:typed_data';

import 'package:record/record.dart';

import 'preprocessor.dart';

/// Raw PCM16 capture plus the rate it was captured at.
class CaptureResult {
  const CaptureResult({required this.pcmBytes, required this.rate});
  final Uint8List pcmBytes;
  final int rate;
}

/// Records the FULL fixed 5.11 s window as mono PCM16 and hands back the raw
/// bytes. The capture is non-cancellable-to-a-partial-score: [cancel] discards
/// the buffer entirely rather than scoring a truncated window.
///
/// Requests [kNativeRate] (16 kHz) by default; the [kDecimateRate] (48 kHz)
/// path is wired for devices that force 48 kHz and is decimated downstream.
class AudioCaptureService {
  AudioCaptureService({AudioRecorder? recorder})
      : _recorder = recorder ?? AudioRecorder();

  final AudioRecorder _recorder;
  bool _cancelled = false;

  Future<bool> hasPermission() => _recorder.hasPermission();

  /// Samples needed for a 5.11 s window at [rate]; after any decimation this is
  /// always [kTargetSamples] at 16 kHz.
  static int targetSamplesFor(int rate) => (kTargetSamples * rate) ~/ kTargetRate;

  /// Abort the in-flight capture; [captureWindow] returns null.
  void cancel() => _cancelled = true;

  /// Capture one 5.11 s window. [onProgress] reports 0..1 fill for the ring.
  /// Returns null if [cancel] was called. Throws [SampleRateException] for an
  /// unsupported [rate].
  Future<CaptureResult?> captureWindow({
    int rate = kNativeRate,
    void Function(double progress)? onProgress,
  }) async {
    resolveRate(rate); // validate up front; never silently resample.
    _cancelled = false;

    final targetBytes = targetSamplesFor(rate) * 2; // PCM16 = 2 bytes/sample
    final builder = BytesBuilder(copy: true);
    final done = Completer<void>();

    final stream = await _recorder.startStream(
      RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: rate,
        numChannels: 1,
        autoGain: false,
        echoCancel: false,
        noiseSuppress: false,
      ),
    );

    late final StreamSubscription<Uint8List> sub;
    void finish() {
      if (!done.isCompleted) done.complete();
    }

    sub = stream.listen(
      (chunk) {
        if (_cancelled) {
          finish();
          return;
        }
        builder.add(chunk);
        onProgress?.call((builder.length / targetBytes).clamp(0.0, 1.0));
        if (builder.length >= targetBytes) finish();
      },
      onError: (_) => finish(),
      onDone: finish,
      cancelOnError: true,
    );

    await done.future;
    await sub.cancel();
    await _recorder.stop();

    if (_cancelled) return null;
    return CaptureResult(pcmBytes: builder.toBytes(), rate: rate);
  }

  Future<void> dispose() => _recorder.dispose();
}
