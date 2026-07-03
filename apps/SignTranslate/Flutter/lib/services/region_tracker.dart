import '../config.dart';
import '../models/text_region.dart';

/// One cached text region: the latest matched quad (upright frame space),
/// the recognized string (null until the recognizer has run on it), and its
/// staleness counter.
class TrackedRegion {
  TrackedRegion(this.id, this.quad);

  final int id;

  /// Updated to the newest matched quad each detection cycle, so the overlay
  /// keeps tracking the sign as the camera moves even when the string is
  /// served from cache.
  Quad quad;

  /// Decoded string; null = recognition still pending. Empty string is a
  /// valid CACHED result ("region decodes to nothing" — GATE-2 ruling #4):
  /// it is displayed outline-only and does NOT consume recognizer budget
  /// again.
  String? text;
  double confidence = 0.0;

  /// Detection cycles since this region last matched a detected quad.
  int missedCycles = 0;

  bool get isRecognized => text != null;
}

/// Outcome of one detection-cycle update: which regions can be displayed from
/// cache and which still need recognizer budget.
class TrackerUpdate {
  const TrackerUpdate({required this.hits, required this.misses});

  /// Matched regions that already carry a recognized string — displayed
  /// WITHOUT invoking the recognizer.
  final List<TrackedRegion> hits;

  /// Regions needing recognition: brand-new quads plus matched regions whose
  /// recognition never completed.
  final List<TrackedRegion> misses;
}

/// IoU-keyed staggered recognition cache (SPEC-mandated behavior #2).
///
/// Each recognized string is keyed to its quad. On subsequent detection
/// cycles, new quads are greedily matched to cached regions by axis-aligned
/// bbox IoU >= [iouThreshold]; hits re-display the cached string without
/// re-running the recognizer, and only misses consume recognizer budget.
/// Regions unmatched for [evictAfterMissedCycles] cycles are evicted.
class RegionTracker {
  RegionTracker({
    this.iouThreshold = kIouCacheThreshold,
    this.evictAfterMissedCycles = kEvictAfterMissedCycles,
  });

  final double iouThreshold;
  final int evictAfterMissedCycles;

  final List<TrackedRegion> _regions = [];
  int _nextId = 0;

  /// All live regions (recognized or pending).
  List<TrackedRegion> get regions => List.unmodifiable(_regions);

  /// Ingests one detection cycle's quads (upright frame space).
  TrackerUpdate update(List<Quad> detected) {
    // Score all candidate (region, quad) pairs above the IoU threshold, then
    // greedily assign best-first so each region and each quad match at most
    // once.
    final pairs = <(double, int, int)>[]; // (iou, regionIdx, quadIdx)
    for (var r = 0; r < _regions.length; r++) {
      for (var q = 0; q < detected.length; q++) {
        final iou = _regions[r].quad.bboxIou(detected[q]);
        if (iou >= iouThreshold) pairs.add((iou, r, q));
      }
    }
    pairs.sort((a, b) => b.$1.compareTo(a.$1));

    final regionMatched = List<bool>.filled(_regions.length, false);
    final quadMatched = List<bool>.filled(detected.length, false);
    final hits = <TrackedRegion>[];
    final misses = <TrackedRegion>[];

    for (final (_, r, q) in pairs) {
      if (regionMatched[r] || quadMatched[q]) continue;
      regionMatched[r] = true;
      quadMatched[q] = true;
      final region = _regions[r]
        ..quad = detected[q]
        ..missedCycles = 0;
      (region.isRecognized ? hits : misses).add(region);
    }

    // Unmatched detected quads become new (pending) regions.
    for (var q = 0; q < detected.length; q++) {
      if (quadMatched[q]) continue;
      final region = TrackedRegion(_nextId++, detected[q]);
      _regions.add(region);
      misses.add(region);
    }

    // Unmatched cached regions go stale; evict the expired.
    for (var r = 0; r < regionMatched.length; r++) {
      if (!regionMatched[r]) _regions[r].missedCycles++;
    }
    _regions.removeWhere((t) => t.missedCycles > evictAfterMissedCycles);

    return TrackerUpdate(hits: hits, misses: misses);
  }

  /// Commits a recognizer result to a region (by identity; the region may
  /// have been evicted between scheduling and completion, which is fine).
  void commit(TrackedRegion region, String text, double confidence) {
    region.text = text;
    region.confidence = confidence;
  }

  /// Whether [region] is still tracked (staggered recognition can outlive a
  /// region's eviction).
  bool isAlive(TrackedRegion region) => _regions.contains(region);

  /// Display list for the overlay: recognized regions matched in the LATEST
  /// detection cycle (missedCycles == 0). Stale entries stay cached for
  /// re-matching but are not drawn — no ghost overlays.
  List<RecognizedRegion> displayRegions() => [
        for (final t in _regions)
          if (t.isRecognized && t.missedCycles == 0)
            RecognizedRegion(
              quad: t.quad,
              text: t.text!,
              confidence: t.confidence,
              fromCache: true,
            ),
      ];

  void clear() {
    _regions.clear();
  }
}
