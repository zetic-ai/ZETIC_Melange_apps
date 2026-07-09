import 'dart:async';

enum PacingMode {
  /// Next tick starts only after the previous finishes → measured FPS is the true
  /// sustainable pipeline throughput.
  benchmark,

  /// Ticks paced to the clip's fps (mirrors a live stream). Late ticks don't wait.
  realtime,
}

/// A pure pacer: repeatedly invokes [onTick] with the chosen pacing. The caller
/// owns the frame index and the active clip, so switching clips is race-free
/// (no per-clip loop to tear down). Ported from Pose-Motion's ClipFrameSource.
class ClipSource {
  ClipSource({required this.fps, this.mode = PacingMode.benchmark});

  final double fps;
  PacingMode mode;

  /// Awaited each tick — do preprocess + inference + render here.
  Future<void> Function()? onTick;

  bool _running = false;
  bool get isRunning => _running;

  Future<void> start() async {
    if (_running) return;
    _running = true;
    final frameIntervalUs = (1e6 / fps).round();
    while (_running) {
      final sw = Stopwatch()..start();
      final cb = onTick;
      if (cb != null) await cb();

      if (mode == PacingMode.realtime) {
        final waitUs = frameIntervalUs - sw.elapsedMicroseconds;
        if (waitUs > 0) {
          await Future<void>.delayed(Duration(microseconds: waitUs));
        }
      } else {
        // Yield so the UI isolate can paint between frames.
        await Future<void>.delayed(Duration.zero);
      }
    }
  }

  void stop() => _running = false;
}
