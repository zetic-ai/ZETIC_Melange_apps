import 'dart:collection';

/// Immutable snapshot rendered by the HUD.
class BenchmarkSnapshot {
  const BenchmarkSnapshot({
    this.inferenceMs = 0,
    this.inferenceP95Ms = 0,
    this.preprocessMs = 0,
    this.totalMs = 0,
    this.fps = 0,
    this.memoryMB = 0,
    this.peakMemoryMB = 0,
    this.meanIoU = 0,
    this.frames = 0,
  });
  final double inferenceMs, inferenceP95Ms, preprocessMs, totalMs;
  final double fps;
  final double memoryMB, peakMemoryMB;
  final double meanIoU;
  final int frames;
}

/// Rolling per-stage latency + 2s-window sustained FPS + peak memory + mean IoU.
/// Ported from apps/Pose-Motion/iOS/PoseMotion/Benchmark/BenchmarkStats.swift.
class BenchmarkStats {
  final _inference = _Rolling(120);
  final _preprocess = _Rolling(120);
  final _total = _Rolling(120);
  final _iou = _Rolling(120);
  final Queue<double> _stamps = Queue<double>(); // seconds
  double _peakMemoryMB = 0;
  double _memoryMB = 0;
  int _frames = 0;

  final _clock = Stopwatch()..start();
  double get _now => _clock.elapsedMicroseconds / 1e6;

  void record({
    required double inferenceMs,
    required double preprocessMs,
    required double iou,
    required double memoryMB,
  }) {
    _inference.push(inferenceMs);
    _preprocess.push(preprocessMs);
    _total.push(inferenceMs + preprocessMs);
    _iou.push(iou);
    _frames++;

    final now = _now;
    _stamps.addLast(now);
    while (_stamps.isNotEmpty && now - _stamps.first > 2.0) {
      _stamps.removeFirst();
    }

    _memoryMB = memoryMB;
    if (memoryMB > _peakMemoryMB) _peakMemoryMB = memoryMB;
  }

  BenchmarkSnapshot snapshot() {
    double fps = 0;
    if (_stamps.length > 1) {
      final span = _now - _stamps.first;
      if (span > 0.2) fps = (_stamps.length - 1) / span;
    }
    return BenchmarkSnapshot(
      inferenceMs: _inference.mean,
      inferenceP95Ms: _inference.p95,
      preprocessMs: _preprocess.mean,
      totalMs: _total.mean,
      fps: fps,
      memoryMB: _memoryMB,
      peakMemoryMB: _peakMemoryMB,
      meanIoU: _iou.mean,
      frames: _frames,
    );
  }
}

class _Rolling {
  _Rolling(this.capacity);
  final int capacity;
  final List<double> _v = [];
  void push(double x) {
    _v.add(x);
    if (_v.length > capacity) _v.removeAt(0);
  }

  double get mean => _v.isEmpty ? 0 : _v.reduce((a, b) => a + b) / _v.length;

  double get p95 {
    if (_v.isEmpty) return 0;
    final s = List<double>.from(_v)..sort();
    final idx = ((s.length - 1) * 0.95).round();
    return s[idx];
  }
}
