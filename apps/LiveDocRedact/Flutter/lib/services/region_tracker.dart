import 'dart:ui';

import '../models/read_field.dart';
import '../models/text_region.dart';

/// Default per-frame recognizer budget. Tuned for the realistic CPU fallback
/// (~32 ms per recognizer crop); raise once `runtimeApType=NPU` is confirmed
/// on the device console.
const int kRecognizerBudgetK = 3;

/// Intersection-over-union of two axis-aligned rects.
double iou(Rect a, Rect b) {
  final double ix = (a.right < b.right ? a.right : b.right) -
      (a.left > b.left ? a.left : b.left);
  if (ix <= 0) return 0;
  final double iy = (a.bottom < b.bottom ? a.bottom : b.bottom) -
      (a.top > b.top ? a.top : b.top);
  if (iy <= 0) return 0;
  final double inter = ix * iy;
  final double union =
      a.width * a.height + b.width * b.height - inter;
  return union <= 0 ? 0 : inter / union;
}

/// A text region tracked across frames, carrying its cached recognition and
/// PII state so already-read fields keep their redaction without re-running
/// the recognizer.
class TrackedRegion {
  TrackedRegion({
    required this.id,
    required this.quad,
    required this.bbox,
    required this.score,
  });

  final int id;
  List<Offset> quad;
  Rect bbox;
  double score;

  /// null until the recognizer has read this region.
  String? text;
  double textConfidence = 0;
  PiiClass piiClass = PiiClass.other;

  int framesUnseen = 0;
  int lastScheduledFrame = -1;

  bool get isRead => text != null;
}

/// The recognizer-budget scheduler (SPEC-binding): staggered recognition with
/// an IoU-keyed cache plus a hard top-K per-frame cap.
///
/// Each frame:
/// - [update] IoU-matches the new detector regions to tracked regions
///   (greedy, highest IoU first, threshold [iouThreshold]); matched regions
///   refresh their geometry and keep their cached text/PII, unmatched
///   detections become new tracks, and tracks unseen for more than
///   [maxUnseenFrames] frames expire.
/// - [scheduleRecognition] returns at most K regions to run the recognizer
///   on: never-read regions first (largest area first — big fields carry the
///   headline PII), then already-read regions whose last read is at least
///   [rereadIntervalFrames] frames old (stalest first). Already-read regions
///   are never rescheduled inside that interval — the cache is reused.
class RegionTracker {
  RegionTracker({
    this.iouThreshold = 0.5,
    this.maxUnseenFrames = 8,
    this.rereadIntervalFrames = 45,
  });

  final double iouThreshold;
  final int maxUnseenFrames;
  final int rereadIntervalFrames;

  final List<TrackedRegion> _tracked = [];
  int _frame = 0;
  int _nextId = 0;

  List<TrackedRegion> get tracked => List.unmodifiable(_tracked);

  int get frame => _frame;

  /// Matches this frame's detections against the tracked set.
  void update(List<TextRegion> detections) {
    _frame++;

    // Greedy IoU matching, highest IoU first.
    final candidates = <(double, int, int)>[]; // (iou, detIdx, trackIdx)
    for (var d = 0; d < detections.length; d++) {
      for (var t = 0; t < _tracked.length; t++) {
        final double v = iou(detections[d].bbox, _tracked[t].bbox);
        if (v >= iouThreshold) candidates.add((v, d, t));
      }
    }
    candidates.sort((a, b) => b.$1.compareTo(a.$1));

    final detMatched = List<bool>.filled(detections.length, false);
    final trackMatched = List<bool>.filled(_tracked.length, false);
    for (final (_, d, t) in candidates) {
      if (detMatched[d] || trackMatched[t]) continue;
      detMatched[d] = true;
      trackMatched[t] = true;
      final region = detections[d];
      final track = _tracked[t];
      track
        ..quad = region.quad
        ..bbox = region.bbox
        ..score = region.score
        ..framesUnseen = 0;
    }

    for (var t = 0; t < _tracked.length; t++) {
      if (!trackMatched[t]) _tracked[t].framesUnseen++;
    }
    _tracked.removeWhere((r) => r.framesUnseen > maxUnseenFrames);

    for (var d = 0; d < detections.length; d++) {
      if (detMatched[d]) continue;
      final region = detections[d];
      _tracked.add(TrackedRegion(
        id: _nextId++,
        quad: region.quad,
        bbox: region.bbox,
        score: region.score,
      ));
    }
  }

  /// Picks at most [k] regions to recognize this frame and marks them
  /// scheduled (so staggering advances even before results arrive).
  List<TrackedRegion> scheduleRecognition(int k) {
    if (k <= 0) return const [];

    final unread = <TrackedRegion>[];
    final rereadable = <TrackedRegion>[];
    for (final r in _tracked) {
      if (r.framesUnseen > 0) continue; // only recognize currently-visible
      if (!r.isRead) {
        if (r.lastScheduledFrame < 0 || r.lastScheduledFrame < _frame) {
          unread.add(r);
        }
      } else if (_frame - r.lastScheduledFrame >= rereadIntervalFrames) {
        rereadable.add(r);
      }
    }

    // Unread: largest area first. Rereads: stalest first.
    unread.sort((a, b) => (b.bbox.width * b.bbox.height)
        .compareTo(a.bbox.width * a.bbox.height));
    rereadable.sort(
        (a, b) => a.lastScheduledFrame.compareTo(b.lastScheduledFrame));

    final picked = <TrackedRegion>[];
    for (final r in unread) {
      if (picked.length >= k) break;
      picked.add(r);
    }
    for (final r in rereadable) {
      if (picked.length >= k) break;
      picked.add(r);
    }
    for (final r in picked) {
      r.lastScheduledFrame = _frame;
    }
    return picked;
  }

  /// Stores a recognition result in the cache.
  void applyRecognition(int id, String text, double confidence) {
    for (final r in _tracked) {
      if (r.id == id) {
        r.text = text;
        r.textConfidence = confidence;
        return;
      }
    }
  }
}
