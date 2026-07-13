import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:livedocredact/models/text_region.dart';
import 'package:livedocredact/services/region_tracker.dart';

TextRegion region(Rect r, {double score = 0.9}) => TextRegion(
      quad: [
        Offset(r.left, r.top),
        Offset(r.right, r.top),
        Offset(r.right, r.bottom),
        Offset(r.left, r.bottom),
      ],
      bbox: r,
      score: score,
    );

/// N stacked fields, each 200x30, 20px apart, with growing width so area
/// ordering is deterministic (field i is wider than field i+1).
List<TextRegion> stackedFields(int n) => [
      for (var i = 0; i < n; i++)
        region(Rect.fromLTWH(10, 10 + i * 50.0, 300.0 - i * 10, 30)),
    ];

void main() {
  test('iou helper: exact overlap 1, disjoint 0, half overlap computed', () {
    const a = Rect.fromLTWH(0, 0, 100, 100);
    expect(iou(a, a), closeTo(1.0, 1e-9));
    expect(iou(a, const Rect.fromLTWH(200, 200, 10, 10)), 0.0);
    expect(iou(a, const Rect.fromLTWH(50, 0, 100, 100)),
        closeTo(5000 / 15000, 1e-9));
  });

  test('K-cap: never more than K regions scheduled per frame', () {
    final tracker = RegionTracker();
    tracker.update(stackedFields(10));
    expect(tracker.scheduleRecognition(3), hasLength(3));
    expect(tracker.tracked, hasLength(10));
  });

  test('unread regions are prioritized largest-area first', () {
    final tracker = RegionTracker();
    tracker.update(stackedFields(5));
    final picked = tracker.scheduleRecognition(2);
    // Fields 0 and 1 are the widest (largest area).
    expect(picked[0].bbox.width, 300);
    expect(picked[1].bbox.width, 290);
  });

  test('IoU-matched read regions keep cached text and are NOT rescheduled',
      () {
    final tracker = RegionTracker();
    final fields = stackedFields(3);
    tracker.update(fields);
    final first = tracker.scheduleRecognition(3);
    expect(first, hasLength(3));
    for (final r in first) {
      tracker.applyRecognition(r.id, 'TEXT-${r.id}', 0.9);
    }

    // Same regions, slightly shifted (IoU still >= 0.5).
    final shifted = [
      for (final f in fields)
        region(f.bbox.shift(const Offset(6, 3)), score: 0.9),
    ];
    tracker.update(shifted);

    expect(tracker.tracked, hasLength(3), reason: 'matched, not duplicated');
    for (final r in tracker.tracked) {
      expect(r.isRead, isTrue, reason: 'cache must survive the match');
      expect(r.text, 'TEXT-${r.id}');
      expect(r.bbox.left, closeTo(16, 1e-9),
          reason: 'geometry refreshes to the new detection');
    }
    expect(tracker.scheduleRecognition(3), isEmpty,
        reason: 'already-read regions are not re-recognized inside the '
            're-read interval — the cache is reused');
  });

  test('staggering covers all regions across frames under a K budget', () {
    final tracker = RegionTracker();
    final fields = stackedFields(10);
    var framesNeeded = 0;
    for (var frame = 0; frame < 10; frame++) {
      tracker.update(fields);
      final picked = tracker.scheduleRecognition(3);
      for (final r in picked) {
        tracker.applyRecognition(r.id, 'T${r.id}', 0.9);
      }
      framesNeeded++;
      if (tracker.tracked.every((r) => r.isRead)) break;
    }
    expect(framesNeeded, 4, reason: '10 regions at K=3 -> read in 4 frames');
    expect(tracker.tracked.every((r) => r.isRead), isTrue);
  });

  test('regions expire after maxUnseenFrames, and persist before that', () {
    final tracker = RegionTracker(maxUnseenFrames: 3);
    tracker.update([region(const Rect.fromLTWH(0, 0, 100, 30))]);
    final id = tracker.tracked.single.id;
    tracker.applyRecognition(id, 'CACHED', 0.9);

    // Missed for 3 frames: still tracked (cached redaction persists).
    for (var i = 0; i < 3; i++) {
      tracker.update(const []);
    }
    expect(tracker.tracked, hasLength(1));
    expect(tracker.tracked.single.text, 'CACHED');

    // A 4th missed frame exceeds the budget: expired.
    tracker.update(const []);
    expect(tracker.tracked, isEmpty);
  });

  test('temporarily-hidden regions are not scheduled while unseen', () {
    final tracker = RegionTracker(maxUnseenFrames: 5);
    tracker.update([region(const Rect.fromLTWH(0, 0, 100, 30))]);
    tracker.update(const []); // now unseen
    expect(tracker.scheduleRecognition(3), isEmpty,
        reason: 'no crop can be taken from a frame the region is not in');
  });

  test('read regions become re-readable after the re-read interval', () {
    final tracker = RegionTracker(rereadIntervalFrames: 3);
    final fields = [region(const Rect.fromLTWH(0, 0, 100, 30))];
    tracker.update(fields);
    final picked = tracker.scheduleRecognition(1);
    tracker.applyRecognition(picked.single.id, 'V1', 0.9);

    tracker.update(fields);
    expect(tracker.scheduleRecognition(1), isEmpty, reason: 'inside interval');
    tracker.update(fields);
    expect(tracker.scheduleRecognition(1), isEmpty, reason: 'inside interval');
    tracker.update(fields);
    expect(tracker.scheduleRecognition(1), hasLength(1),
        reason: 'stale read is refreshed after the interval');
  });

  test('unread regions outrank stale re-reads for the budget', () {
    final tracker = RegionTracker(rereadIntervalFrames: 1);
    final a = region(const Rect.fromLTWH(0, 0, 100, 30));
    tracker.update([a]);
    final picked = tracker.scheduleRecognition(1);
    tracker.applyRecognition(picked.single.id, 'OLD', 0.9);

    // A new unread region appears; budget K=1 must go to it, not the re-read.
    final b = region(const Rect.fromLTWH(0, 200, 100, 30));
    tracker.update([a, b]);
    final next = tracker.scheduleRecognition(1);
    expect(next, hasLength(1));
    expect(next.single.isRead, isFalse);
  });

  test('non-overlapping detection becomes a NEW track (no cache leak)', () {
    final tracker = RegionTracker();
    tracker.update([region(const Rect.fromLTWH(0, 0, 100, 30))]);
    final id = tracker.tracked.single.id;
    tracker.applyRecognition(id, 'SECRET', 0.9);

    // A far-away region must not inherit the cached text.
    tracker.update([
      region(const Rect.fromLTWH(0, 0, 100, 30)),
      region(const Rect.fromLTWH(400, 400, 100, 30)),
    ]);
    expect(tracker.tracked, hasLength(2));
    final fresh = tracker.tracked.firstWhere((r) => r.id != id);
    expect(fresh.isRead, isFalse);
  });
}
