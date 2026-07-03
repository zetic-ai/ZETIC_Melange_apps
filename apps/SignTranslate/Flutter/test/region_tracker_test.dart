import 'dart:ui' show Rect;

import 'package:flutter_test/flutter_test.dart';
import 'package:signtranslate/config.dart';
import 'package:signtranslate/models/text_region.dart';
import 'package:signtranslate/services/region_tracker.dart';

Quad rectQuad(double l, double t, double w, double h) =>
    Quad.fromRect(Rect.fromLTWH(l, t, w, h));

void main() {
  test('IoU cache threshold is SPEC-exact: 0.5', () {
    expect(kIouCacheThreshold, 0.5);
  });

  group('IoU-keyed cache hits (mandated behavior 2)', () {
    test('slightly-moved quad (IoU >= 0.5) reuses the cached string and does '
        'NOT reach the recognizer', () {
      final tracker = RegionTracker();

      // Cycle 1: new region -> miss -> "recognize" it.
      final first = tracker.update([rectQuad(100, 100, 200, 60)]);
      expect(first.misses, hasLength(1));
      expect(first.hits, isEmpty);
      tracker.commit(first.misses.single, 'BAHNHOF', 0.93);

      // Cycle 2: camera moved a little (IoU ~ 0.72).
      final moved = rectQuad(120, 105, 200, 60);
      expect(moved.bboxIou(rectQuad(100, 100, 200, 60)),
          greaterThanOrEqualTo(0.5));
      final second = tracker.update([moved]);

      // Cache HIT: no recognition candidates at all this cycle.
      expect(second.misses, isEmpty,
          reason: 'a hit must not consume recognizer budget');
      expect(second.hits, hasLength(1));
      expect(second.hits.single.text, 'BAHNHOF');

      // The stored quad tracked the motion (overlay follows the sign).
      expect(second.hits.single.quad.tl.dx, 120);
    });

    test('IoU boundary: just-below 0.5 is a miss, just-above is a hit', () {
      // Base 100x100 box at origin; shifted horizontally by dx the IoU is
      // (100-dx)/(100+dx): dx=33 -> 0.503 (hit), dx=34 -> 0.493 (miss).
      final base = rectQuad(0, 0, 100, 100);
      expect(base.bboxIou(rectQuad(33, 0, 100, 100)), greaterThan(0.5));
      expect(base.bboxIou(rectQuad(34, 0, 100, 100)), lessThan(0.5));

      final hitTracker = RegionTracker();
      hitTracker.commit(hitTracker.update([base]).misses.single, 'X', 1);
      expect(hitTracker.update([rectQuad(33, 0, 100, 100)]).hits,
          hasLength(1));

      final missTracker = RegionTracker();
      missTracker.commit(missTracker.update([base]).misses.single, 'X', 1);
      final r = missTracker.update([rectQuad(34, 0, 100, 100)]);
      expect(r.hits, isEmpty);
      expect(r.misses, hasLength(1)); // the new quad needs recognition
    });

    test('a matched-but-never-recognized region stays a miss (still pending)',
        () {
      final tracker = RegionTracker();
      tracker.update([rectQuad(0, 0, 100, 50)]); // miss, NOT committed
      final again = tracker.update([rectQuad(2, 0, 100, 50)]);
      expect(again.misses, hasLength(1));
      expect(again.hits, isEmpty);
    });

    test('empty-string results are cached too (ruling #4): a false-positive '
        'region does not churn recognizer budget', () {
      final tracker = RegionTracker();
      final first = tracker.update([rectQuad(0, 0, 100, 50)]);
      tracker.commit(first.misses.single, '', 0);
      final second = tracker.update([rectQuad(1, 0, 100, 50)]);
      expect(second.misses, isEmpty); // cached, no re-run
      expect(second.hits.single.text, '');
    });
  });

  group('new regions and greedy matching', () {
    test('a brand-new quad becomes a pending region (miss)', () {
      final tracker = RegionTracker();
      tracker.commit(
          tracker.update([rectQuad(0, 0, 100, 50)]).misses.single, 'A', 1);
      final r = tracker.update([
        rectQuad(0, 0, 100, 50), // known
        rectQuad(400, 400, 120, 40), // new
      ]);
      expect(r.hits, hasLength(1));
      expect(r.misses, hasLength(1));
      expect(r.misses.single.quad.tl.dx, 400);
    });

    test('each cached region matches at most one quad (greedy best-first)',
        () {
      final tracker = RegionTracker();
      tracker.commit(
          tracker.update([rectQuad(0, 0, 100, 100)]).misses.single, 'A', 1);
      // Two candidates BOTH above the IoU threshold against the one cached
      // region (0.905 and 0.667): greedy best-first gives the match to the
      // closer quad; the other becomes a new region.
      final r = tracker.update([
        rectQuad(5, 0, 100, 100), // IoU ~0.905
        rectQuad(20, 0, 100, 100), // IoU ~0.667
      ]);
      expect(r.hits, hasLength(1));
      expect(r.hits.single.quad.tl.dx, 5);
      expect(r.misses, hasLength(1));
      expect(r.misses.single.quad.tl.dx, 20);
    });
  });

  group('eviction (mandated: evict stale entries)', () {
    test('a region unmatched for N cycles is evicted; the same quad later '
        'is a fresh miss again', () {
      final tracker = RegionTracker(evictAfterMissedCycles: 3);
      final quad = rectQuad(0, 0, 100, 50);
      tracker.commit(tracker.update([quad]).misses.single, 'HELLO', 1);
      expect(tracker.regions, hasLength(1));

      // 3 empty cycles: still cached (missedCycles 1,2,3 <= 3).
      for (var i = 0; i < 3; i++) {
        tracker.update([]);
      }
      expect(tracker.regions, hasLength(1));

      // 4th empty cycle: evicted.
      tracker.update([]);
      expect(tracker.regions, isEmpty);

      // Re-appearing quad is a miss (recognizer must run again).
      expect(tracker.update([quad]).misses, hasLength(1));
    });

    test('a matched region resets its staleness counter', () {
      final tracker = RegionTracker(evictAfterMissedCycles: 2);
      final quad = rectQuad(0, 0, 100, 50);
      tracker.commit(tracker.update([quad]).misses.single, 'X', 1);
      tracker.update([]); // missed once
      tracker.update([quad]); // matched -> reset
      tracker.update([]);
      tracker.update([]);
      expect(tracker.regions, hasLength(1)); // 2 misses <= 2, still alive
    });

    test('default eviction horizon matches config', () {
      expect(RegionTracker().evictAfterMissedCycles, kEvictAfterMissedCycles);
    });
  });

  group('display list', () {
    test('only currently-matched, recognized regions display (no ghosts, '
        'no unrecognized outlines from stale cycles)', () {
      final tracker = RegionTracker();
      final a = rectQuad(0, 0, 100, 50);
      final b = rectQuad(300, 300, 100, 50);
      final first = tracker.update([a, b]);
      tracker.commit(first.misses[0], 'ALPHA', 0.9);
      // b never recognized.

      var display = tracker.displayRegions();
      expect(display, hasLength(1));
      expect(display.single.text, 'ALPHA');
      expect(display.single.fromCache, isTrue);

      // a vanishes: cached but NOT displayed.
      tracker.update([b]);
      display = tracker.displayRegions();
      expect(display, isEmpty);
    });
  });
}
