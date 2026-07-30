import 'dart:ui' show Rect;

import 'package:flutter_test/flutter_test.dart';
import 'package:signtranslate/config.dart';
import 'package:signtranslate/models/text_region.dart';
import 'package:signtranslate/services/frame_scheduler.dart';
import 'package:signtranslate/services/region_tracker.dart';

TrackedRegion region(int id, double l, double t, double w, double h) =>
    TrackedRegion(id, Quad.fromRect(Rect.fromLTWH(l, t, w, h)));

void main() {
  test('scheduler defaults are SPEC-exact: K=3, duty 0.5', () {
    expect(kTopK, 3);
    expect(kDetectionDutyTarget, 0.5);
    final s = FrameScheduler();
    expect(s.topK, 3);
  });

  group('top-K selection (mandated behavior 1)', () {
    test('caps at K per frame: 10 candidates -> exactly 3 selected', () {
      final s = FrameScheduler();
      final misses = [
        for (var i = 0; i < 10; i++) region(i, 0, i * 100.0, 50 + i * 10.0, 20),
      ];
      expect(s.selectForRecognition(misses), hasLength(3));
    });

    test('priority: LARGEST area first (nearest signs)', () {
      final s = FrameScheduler();
      final small = region(0, 0, 0, 40, 20); // 800
      final big = region(1, 0, 0, 300, 80); // 24000
      final mid = region(2, 0, 0, 100, 50); // 5000
      final tiny = region(3, 0, 0, 10, 10); // 100
      final picked = s.selectForRecognition([small, big, mid, tiny]);
      expect(picked, [big, mid, small]);
      expect(picked, isNot(contains(tiny)));
    });

    test('fewer than K candidates: all selected, none invented', () {
      final s = FrameScheduler();
      expect(s.selectForRecognition([region(0, 0, 0, 10, 10)]), hasLength(1));
      expect(s.selectForRecognition([]), isEmpty);
    });

    test('cache-misses-first is structural: only misses ever reach '
        'selection (hits are served by the tracker, bypassing budget)', () {
      // The API takes the tracker's miss list; RegionTracker tests prove
      // hits never appear in it. Here: an explicit limit override supports
      // draining the staggered queue.
      final s = FrameScheduler();
      final misses = [for (var i = 0; i < 5; i++) region(i, 0, 0, 10.0 + i, 10)];
      expect(s.selectForRecognition(misses, limit: 5), hasLength(5));
    });
  });

  group('adaptive detection cadence with a fake clock (mandated behavior 3)',
      () {
    test('first frame always detects', () {
      var now = 0;
      final s = FrameScheduler(clock: () => now);
      expect(s.shouldRunDetection(), isTrue);
    });

    test('fast passes (NPU case, <= 33 ms) detect EVERY frame', () {
      var now = 0;
      final s = FrameScheduler(clock: () => now);
      s.recordDetectionPass(passMs: 12, modelMs: 8);
      now += 1; // the very next millisecond
      expect(s.shouldRunDetection(), isTrue);
    });

    test('slow passes (CPU fallback ~169 ms) are spaced to ~50% duty', () {
      var now = 0;
      final s = FrameScheduler(clock: () => now);
      // Simulate the detection pass taking 169 ms of wall time.
      now = 169;
      s.recordDetectionPass(passMs: 169, modelMs: 169);

      // Immediately after (and up to +168 ms): gated off.
      now = 170;
      expect(s.shouldRunDetection(), isFalse);
      now = 169 + 168;
      expect(s.shouldRunDetection(), isFalse);

      // At +169 ms (one pass length of cool-down = 50% duty): allowed.
      now = 169 + 169;
      expect(s.shouldRunDetection(), isTrue);
    });

    test('the interval ADAPTS as measured latency changes', () {
      var now = 0;
      final s = FrameScheduler(clock: () => now);
      s.recordDetectionPass(passMs: 400, modelMs: 380);
      // EMA = 400 -> wait 400 ms.
      now = 200;
      expect(s.shouldRunDetection(), isFalse);
      now = 400;
      expect(s.shouldRunDetection(), isTrue);

      // Backend got fast (say the NPU artifact arrived): EMA decays toward
      // fast passes and cadence opens up.
      for (var i = 0; i < 12; i++) {
        s.recordDetectionPass(passMs: 10, modelMs: 6);
      }
      expect(s.emaDetectionPassMs, lessThan(kFrameBudgetMs));
      now += 1;
      expect(s.shouldRunDetection(), isTrue);
    });

    test('cool-down is capped at kMaxDetectionIntervalMs', () {
      var now = 0;
      final s = FrameScheduler(clock: () => now);
      s.recordDetectionPass(passMs: 100000, modelMs: 100000);
      now = kMaxDetectionIntervalMs;
      expect(s.shouldRunDetection(), isTrue);
    });
  });

  group('busy guard / frame dropping (mandated behavior 4)', () {
    test('frames arriving while a pass is in flight are DROPPED, not queued',
        () {
      final s = FrameScheduler();
      expect(s.tryBeginPass(), isTrue);
      // Three more frames arrive mid-pass: all dropped.
      expect(s.tryBeginPass(), isFalse);
      expect(s.tryBeginPass(), isFalse);
      expect(s.tryBeginPass(), isFalse);
      expect(s.droppedFrames, 3);

      s.endPass();
      expect(s.tryBeginPass(), isTrue); // next frame processes normally
      expect(s.droppedFrames, 3); // and was not counted as dropped
    });

    test('isBusy reflects the in-flight pass', () {
      final s = FrameScheduler();
      expect(s.isBusy, isFalse);
      s.tryBeginPass();
      expect(s.isBusy, isTrue);
      s.endPass();
      expect(s.isBusy, isFalse);
    });
  });

  group('HUD stat recording (mandated behavior 5 feeds from here)', () {
    test('EMAs seed on first sample then smooth', () {
      final s = FrameScheduler();
      s.recordDetectionPass(passMs: 200, modelMs: 170);
      expect(s.emaDetectorModelMs, 170);
      s.recordDetectionPass(passMs: 100, modelMs: 100);
      expect(s.emaDetectorModelMs, closeTo(170 * 0.7 + 100 * 0.3, 1e-9));

      s.recordRecognition(32);
      expect(s.emaRecognizerModelMs, 32);
      s.recordRecognition(16);
      expect(s.emaRecognizerModelMs, closeTo(32 * 0.7 + 16 * 0.3, 1e-9));
    });
  });
}
