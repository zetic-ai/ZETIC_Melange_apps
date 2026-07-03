import 'dart:math' as math;

/// Which demo signal the feed produces.
enum FeedMode {
  /// License-clean, app-generated "industrial machine temperature" replay:
  /// statistically similar to a real 5-min plant sensor (baseline + slow
  /// drift + two cyclic components + noise + a scripted failure arc per
  /// loop). No third-party data is bundled (GATE-2 ruling: NAB is AGPL-3.0
  /// and stays a local Stage-0 validation artifact only).
  industrial,

  /// Clean two-tone sine + noise "lab signal" — the crowd-pleaser mode where
  /// injected anomalies pop unmistakably.
  lab,
}

/// Kinds of user-triggered anomaly injections.
enum InjectionKind { spike, levelShift, noiseBurst }

class _Injection {
  _Injection(this.kind, this.remaining);
  final InjectionKind kind;
  int remaining;
}

/// Deterministic (seeded) sensor-signal generator.
///
/// Every sample is produced by [next]; the same seed + mode + injection
/// sequence reproduces the identical series bit-for-bit (unit-tested), which
/// keeps the demo rehearsable.
class SensorFeed {
  SensorFeed({this.mode = FeedMode.industrial, int seed = 7})
      : _rng = math.Random(seed);

  final FeedMode mode;
  final math.Random _rng;

  int _i = 0;
  final List<_Injection> _active = [];

  // Box-Muller cache.
  double? _spareGauss;

  /// Global index of the NEXT sample to be produced.
  int get nextIndex => _i;

  double _gauss() {
    final spare = _spareGauss;
    if (spare != null) {
      _spareGauss = null;
      return spare;
    }
    double u1 = _rng.nextDouble();
    while (u1 <= 1e-12) {
      u1 = _rng.nextDouble();
    }
    final u2 = _rng.nextDouble();
    final r = math.sqrt(-2.0 * math.log(u1));
    _spareGauss = r * math.sin(2 * math.pi * u2);
    return r * math.cos(2 * math.pi * u2);
  }

  /// Queues an anomaly to corrupt upcoming samples:
  ///  - spike: one sample jumps hard out of band,
  ///  - levelShift: +offset held for 60 samples (process shift),
  ///  - noiseBurst: violent extra noise for 40 samples (instability).
  void inject(InjectionKind kind) {
    switch (kind) {
      case InjectionKind.spike:
        _active.add(_Injection(kind, 1));
      case InjectionKind.levelShift:
        _active.add(_Injection(kind, 60));
      case InjectionKind.noiseBurst:
        _active.add(_Injection(kind, 40));
    }
  }

  /// The scripted failure arc of the industrial loop: sample-index ranges
  /// (within the loop) where the "machine" degrades, so a booth demo has a
  /// built-in story without pressing any button.
  static const int loopLength = 6000;
  static const int failureStart = 4200;
  static const int failureEnd = 4600;

  double _industrial(int i) {
    final p = i % loopLength;
    var v = 88.0 +
        4.0 * math.sin(2 * math.pi * p / 2400.0) + // slow ambient drift
        2.5 * math.sin(2 * math.pi * p / 600.0) + // duty cycle
        1.2 * math.sin(2 * math.pi * p / 140.0) + // fast process cycle
        0.6 * _gauss();
    // Scripted failure: ramp up, oscillate violently, then collapse (trip).
    if (p >= failureStart && p < failureEnd) {
      final k = (p - failureStart) / (failureEnd - failureStart);
      v += 14.0 * k + 3.5 * math.sin(2 * math.pi * p / 9.0) * k;
    } else if (p >= failureEnd && p < failureEnd + 120) {
      v -= 18.0 * (1.0 - (p - failureEnd) / 120.0); // post-trip cooldown
    }
    return v;
  }

  double _lab(int i) {
    return 50.0 +
        8.0 * math.sin(2 * math.pi * i / 96.0) +
        3.0 * math.sin(2 * math.pi * i / 24.0) +
        0.6 * _gauss();
  }

  /// Produces the next sample (base signal + any active injections).
  double next() {
    var v = switch (mode) {
      FeedMode.industrial => _industrial(_i),
      FeedMode.lab => _lab(_i),
    };
    for (final inj in _active) {
      switch (inj.kind) {
        case InjectionKind.spike:
          v += 25.0;
        case InjectionKind.levelShift:
          v += 12.0;
        case InjectionKind.noiseBurst:
          v += 2.5 * _gauss() * 3.0;
      }
      inj.remaining--;
    }
    _active.removeWhere((inj) => inj.remaining <= 0);
    _i++;
    return v;
  }
}
