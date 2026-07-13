import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:livedocredact/services/db_postprocessor.dart';
import 'package:livedocredact/services/detector_preprocessor.dart';

const int kSize = kDetInputSize;

/// Identity letterbox: model space == source space.
const identity = LetterboxGeometry(
    scale: 1, padX: 0, padY: 0, srcWidth: kSize, srcHeight: kSize);

Float32List makeHeatmap() => Float32List(kSize * kSize);

void fillRect(Float32List map, int x0, int y0, int x1, int y1, double v) {
  for (var y = y0; y <= y1; y++) {
    for (var x = x0; x <= x1; x++) {
      map[y * kSize + x] = v;
    }
  }
}

/// The unclip offset DBPostProcess applies to a w-by-h rect.
double unclipDelta(double w, double h, {double ratio = 1.5}) =>
    (w * h) * ratio / (2 * (w + h));

void main() {
  test('single blob -> one quad ~= blob rect + unclip, in PIXEL space', () {
    final map = makeHeatmap();
    fillRect(map, 100, 300, 199, 339, 0.9);
    final result = decodeDbHeatmap(map, identity);

    expect(result.regions, hasLength(1));
    final region = result.regions.single;

    // Extents of the pixel-center rect are 99 x 39.
    final d = unclipDelta(99, 39);
    expect(region.bbox.left, closeTo(100 - d, 1.5));
    expect(region.bbox.top, closeTo(300 - d, 1.5));
    expect(region.bbox.right, closeTo(199 + d, 1.5));
    expect(region.bbox.bottom, closeTo(339 + d, 1.5));

    // Pixel space, not normalized: values far above 1.
    expect(region.bbox.right, greaterThan(2.0));
    expect(region.quad, hasLength(4));
  });

  test('score is the RAW mean heatmap value — no extra sigmoid', () {
    final map = makeHeatmap();
    fillRect(map, 100, 300, 199, 339, 0.9);
    final region = decodeDbHeatmap(map, identity).regions.single;
    // A spurious sigmoid would report sigmoid(0.9) ~= 0.71.
    expect(region.score, closeTo(0.9, 1e-6));
  });

  test('box_thresh boundary: mean 0.59 dropped, 0.61 kept', () {
    final low = makeHeatmap();
    fillRect(low, 100, 300, 199, 339, 0.59);
    expect(decodeDbHeatmap(low, identity).regions, isEmpty);

    final high = makeHeatmap();
    fillRect(high, 100, 300, 199, 339, 0.61);
    expect(decodeDbHeatmap(high, identity).regions, hasLength(1));
  });

  test('binarize threshold boundary: 0.31 joins a component, 0.29 does not',
      () {
    // Core blob at 0.9 plus a 6px-wide right extension just above/below the
    // 0.3 binarization threshold. The extension is small enough to keep the
    // mean above box_thresh either way — only binarization decides.
    final withExt = makeHeatmap();
    fillRect(withExt, 100, 300, 199, 339, 0.9);
    fillRect(withExt, 200, 300, 205, 339, 0.31);
    final wide = decodeDbHeatmap(withExt, identity).regions.single;

    final withoutExt = makeHeatmap();
    fillRect(withoutExt, 100, 300, 199, 339, 0.9);
    fillRect(withoutExt, 200, 300, 205, 339, 0.29);
    final narrow = decodeDbHeatmap(withoutExt, identity).regions.single;

    expect(wide.bbox.width, greaterThan(narrow.bbox.width + 4),
        reason: '0.31 pixels binarize in and widen the box; 0.29 do not');
  });

  test('two adjacent fields stay TWO regions (unclip must not merge them)',
      () {
    final map = makeHeatmap();
    fillRect(map, 100, 300, 149, 320, 0.9);
    fillRect(map, 160, 300, 209, 320, 0.9); // 10px gap
    final regions = decodeDbHeatmap(map, identity).regions;
    expect(regions, hasLength(2),
        reason: 'adjacent name/DOB fields must not fuse into one redaction');
  });

  test('one contiguous word stays ONE region (no split)', () {
    final map = makeHeatmap();
    fillRect(map, 100, 300, 299, 320, 0.9);
    expect(decodeDbHeatmap(map, identity).regions, hasLength(1));
  });

  test('tiny blobs below min side are dropped', () {
    final map = makeHeatmap();
    fillRect(map, 100, 300, 101, 301, 0.95); // 2x2 -> extents 1x1
    expect(decodeDbHeatmap(map, identity).regions, isEmpty);
  });

  test('rotated blob: quad text axis follows the LONG side', () {
    final map = makeHeatmap();
    // Diagonal thick strip: a rotated rectangle-ish component.
    for (var i = 0; i < 120; i++) {
      fillRect(map, 100 + i, 300 + i, 100 + i + 18, 300 + i + 18, 0.9);
    }
    final region = decodeDbHeatmap(map, identity).regions.single;
    final quad = region.quad;
    final top = (quad[1] - quad[0]).distance;
    final side = (quad[3] - quad[0]).distance;
    expect(top, greaterThan(side),
        reason: 'quad[0]->quad[1] must run along the text (long) axis');
    // And it must actually be diagonal: the top edge has a real slope.
    final slope =
        (quad[1].dy - quad[0].dy).abs() / (quad[1].dx - quad[0].dx).abs();
    expect(slope, closeTo(1.0, 0.25));
  });

  test('letterbox inverse round-trip returns a known box to source space', () {
    // Virtual 1280x720 source, scale 0.5, pad (0, 140) — the geometry the
    // real preprocessor produces for 720p frames.
    const geom = LetterboxGeometry(
        scale: 0.5, padX: 0, padY: 140, srcWidth: 1280, srcHeight: 720);
    // Source box [200,100]-[400,160] -> model box [100,190]-[200,220].
    final map = makeHeatmap();
    fillRect(map, 100, 190, 200, 220, 0.9);
    final region = decodeDbHeatmap(map, geom).regions.single;

    final d = unclipDelta(100, 30) / geom.scale; // unclip, in SOURCE pixels
    expect(region.bbox.left, closeTo(200 - d, 3));
    expect(region.bbox.top, closeTo(100 - d, 3));
    expect(region.bbox.right, closeTo(400 + d, 3));
    expect(region.bbox.bottom, closeTo(160 + d, 3));
    // Center round-trips almost exactly (unclip is symmetric).
    expect(region.bbox.center.dx, closeTo(300, 1.5));
    expect(region.bbox.center.dy, closeTo(130, 1.5));
  });

  test('heatmap stats are surfaced for the 0-dB on-device sanity check', () {
    final map = makeHeatmap();
    fillRect(map, 0, 0, 9, 9, 0.8); // 100 px at 0.8
    final stats = decodeDbHeatmap(map, identity).stats;
    expect(stats.min, 0.0);
    expect(stats.max, closeTo(0.8, 1e-6));
    expect(stats.mean, closeTo(0.8 * 100 / (kSize * kSize), 1e-9));
  });

  test('maxCandidates caps the region count', () {
    final map = makeHeatmap();
    // 25 small separate blobs.
    for (var i = 0; i < 25; i++) {
      final x = 20 + (i % 5) * 120;
      final y = 20 + (i ~/ 5) * 120;
      fillRect(map, x, y, x + 20, y + 8, 0.9);
    }
    final capped = decodeDbHeatmap(map, identity,
        params: const DbParams(maxCandidates: 10));
    expect(capped.regions.length, lessThanOrEqualTo(10));
    final all = decodeDbHeatmap(map, identity);
    expect(all.regions, hasLength(25));
  });

  test('regions come back highest-score first (budget priority order)', () {
    final map = makeHeatmap();
    fillRect(map, 100, 100, 199, 130, 0.65);
    fillRect(map, 100, 300, 199, 330, 0.95);
    final regions = decodeDbHeatmap(map, identity).regions;
    expect(regions, hasLength(2));
    expect(regions.first.score, greaterThan(regions.last.score));
  });

  test('quad corners are ordered tl, tr, br, bl for an axis-aligned box', () {
    final map = makeHeatmap();
    fillRect(map, 100, 300, 199, 339, 0.9);
    final quad = decodeDbHeatmap(map, identity).regions.single.quad;
    // tl left of tr, tl above bl; consistent winding.
    expect(quad[0].dx, lessThan(quad[1].dx));
    expect(quad[0].dy, lessThan(quad[3].dy));
    expect(quad[2].dx, greaterThan(quad[3].dx));
    expect(quad[2].dy, greaterThan(quad[1].dy));
    // Roughly a rectangle: opposite sides equal.
    expect((quad[1] - quad[0]).distance,
        closeTo((quad[2] - quad[3]).distance, 1e-3));
    expect(math.min((quad[1] - quad[0]).distance,
            (quad[3] - quad[0]).distance),
        greaterThan(30));
  });
}
