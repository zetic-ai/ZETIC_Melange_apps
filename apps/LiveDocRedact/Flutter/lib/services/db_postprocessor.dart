import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui';

import '../models/text_region.dart';
import 'detector_preprocessor.dart';

/// DBPostProcess parameters (PaddleOCR defaults, SPEC-binding).
class DbParams {
  const DbParams({
    this.thresh = 0.3,
    this.boxThresh = 0.6,
    this.unclipRatio = 1.5,
    this.maxCandidates = 1000,
    this.minSideSize = 3.0,
  });

  /// Binarization threshold on the probability heatmap.
  final double thresh;

  /// Minimum mean heatmap probability over a component for it to become a box.
  final double boxThresh;

  /// Polygon dilation: offset = area * unclipRatio / perimeter.
  final double unclipRatio;

  final int maxCandidates;

  /// Minimum short side (in 640-space pixels, before unclip) of a kept box.
  final double minSideSize;
}

/// Min/max/mean of the raw heatmap — surfaced on the HUD for the Tier-C
/// "dashboard says 0.00 dB" non-degenerate-output sanity check.
class HeatmapStats {
  const HeatmapStats({required this.min, required this.max, required this.mean});

  final double min;
  final double max;
  final double mean;

  @override
  String toString() =>
      'min=${min.toStringAsFixed(3)} max=${max.toStringAsFixed(3)} '
      'mean=${mean.toStringAsFixed(3)}';
}

/// Decoded detector output: text regions in upright source-frame pixels plus
/// heatmap statistics.
class DbDecodeResult {
  const DbDecodeResult({required this.regions, required this.stats});

  final List<TextRegion> regions;
  final HeatmapStats stats;
}

/// Decodes the DBNet probability heatmap ([1,1,640,640] flattened, values
/// already Sigmoid-activated in-graph — NO extra activation is applied here)
/// into text-region quads, then maps them back through the letterbox inverse
/// to upright source-frame pixels.
///
/// Steps (DBPostProcess): binarize at [DbParams.thresh] -> connected
/// components -> drop components whose mean raw probability is below
/// [DbParams.boxThresh] -> min-area rectangle -> unclip (dilate) by
/// [DbParams.unclipRatio] -> inverse letterbox.
DbDecodeResult decodeDbHeatmap(
  Float32List heatmap,
  LetterboxGeometry geometry, {
  DbParams params = const DbParams(),
}) {
  const int size = kDetInputSize;
  const int area = size * size;
  assert(heatmap.length == area);

  // Pass 1: stats + binary mask.
  double mn = double.infinity, mx = double.negativeInfinity, sum = 0;
  final Uint8List mask = Uint8List(area);
  final double thresh = params.thresh;
  for (var i = 0; i < area; i++) {
    final double p = heatmap[i];
    if (p < mn) mn = p;
    if (p > mx) mx = p;
    sum += p;
    if (p > thresh) mask[i] = 1;
  }
  final stats = HeatmapStats(min: mn, max: mx, mean: sum / area);

  // Pass 2: connected components (4-connectivity, iterative flood fill).
  final regions = <TextRegion>[];
  final Int32List stack = Int32List(area);
  // Tier-B: per-row extent tracking uses flat reused Int32Lists instead of a
  // per-component Map (measured 1.91 -> 1.09 ms, -43%, on the A4 DB-decode
  // stage). rowMinX[y] == -1 marks an untouched row; touched rows are reset
  // after each component.
  final Int32List rowMinX = Int32List(size)..fillRange(0, size, -1);
  final Int32List rowMaxX = Int32List(size);
  for (var start = 0; start < area; start++) {
    if (mask[start] != 1) continue;

    // Flood-fill one component; track per-row x extents + mean probability.
    var stackTop = 0;
    stack[stackTop++] = start;
    mask[start] = 2; // visited
    var count = 0;
    var probSum = 0.0;
    var minRow = size, maxRow = -1;

    while (stackTop > 0) {
      final int idx = stack[--stackTop];
      final int y = idx ~/ size;
      final int x = idx - y * size;
      count++;
      probSum += heatmap[idx];
      if (y < minRow) minRow = y;
      if (y > maxRow) maxRow = y;
      final int curMin = rowMinX[y];
      if (curMin == -1) {
        rowMinX[y] = x;
        rowMaxX[y] = x;
      } else {
        if (x < curMin) rowMinX[y] = x;
        if (x > rowMaxX[y]) rowMaxX[y] = x;
      }

      if (x > 0 && mask[idx - 1] == 1) {
        mask[idx - 1] = 2;
        stack[stackTop++] = idx - 1;
      }
      if (x < size - 1 && mask[idx + 1] == 1) {
        mask[idx + 1] = 2;
        stack[stackTop++] = idx + 1;
      }
      if (y > 0 && mask[idx - size] == 1) {
        mask[idx - size] = 2;
        stack[stackTop++] = idx - size;
      }
      if (y < size - 1 && mask[idx + size] == 1) {
        mask[idx + size] = 2;
        stack[stackTop++] = idx + size;
      }
    }

    // Hull candidate points: per-row extremes (this preserves the convex hull
    // of the full pixel set). Collected before the early filters so the
    // reused row arrays are ALWAYS reset for the next component.
    final pts = <Offset>[];
    for (var y = minRow; y <= maxRow; y++) {
      final int a = rowMinX[y];
      if (a == -1) continue;
      final int b = rowMaxX[y];
      pts.add(Offset(a.toDouble(), y.toDouble()));
      if (b != a) pts.add(Offset(b.toDouble(), y.toDouble()));
      rowMinX[y] = -1; // reset for the next component
    }

    if (count < 4) continue; // too small for even a 2x2 blob
    final double meanScore = probSum / count;
    if (meanScore < params.boxThresh) continue;

    final rect = _minAreaRect(pts);
    if (rect == null) continue;
    if (math.min(rect.w, rect.h) < params.minSideSize) continue;

    // Unclip: expand by delta = area * ratio / perimeter (exact polygon offset
    // for a rectangle).
    final double a = rect.w * rect.h;
    final double per = 2 * (rect.w + rect.h);
    final double delta = per > 0 ? a * params.unclipRatio / per : 0;
    final double hw = rect.w / 2 + delta;
    final double hh = rect.h / 2 + delta;

    // Corners in 640-space: tl, tr, br, bl with text axis along u.
    final ux = rect.ux, uy = rect.uy, vx = rect.vx, vy = rect.vy;
    final cx = rect.cx, cy = rect.cy;
    final quad640 = <Offset>[
      Offset(cx - ux * hw - vx * hh, cy - uy * hw - vy * hh),
      Offset(cx + ux * hw - vx * hh, cy + uy * hw - vy * hh),
      Offset(cx + ux * hw + vx * hh, cy + uy * hw + vy * hh),
      Offset(cx - ux * hw + vx * hh, cy - uy * hw + vy * hh),
    ];

    // Inverse letterbox -> upright source-frame pixels.
    final quad = quad640.map(geometry.unmap).toList(growable: false);
    var minX = double.infinity, minY = double.infinity;
    var maxX = double.negativeInfinity, maxY = double.negativeInfinity;
    for (final p in quad) {
      if (p.dx < minX) minX = p.dx;
      if (p.dy < minY) minY = p.dy;
      if (p.dx > maxX) maxX = p.dx;
      if (p.dy > maxY) maxY = p.dy;
    }
    if (maxX - minX < 1 || maxY - minY < 1) continue;

    regions.add(TextRegion(
      quad: quad,
      bbox: Rect.fromLTRB(minX, minY, maxX, maxY),
      score: meanScore,
    ));
    if (regions.length >= params.maxCandidates) break;
  }

  // Highest-score regions first (feeds the recognizer budget priority).
  regions.sort((a, b) => b.score.compareTo(a.score));
  return DbDecodeResult(regions: regions, stats: stats);
}

/// A minimum-area rectangle: center (cx,cy), unit axes u (long/text axis,
/// oriented left-to-right) and v (perpendicular, pointing down), extents w
/// along u and h along v.
class _MinAreaRect {
  const _MinAreaRect(
      this.cx, this.cy, this.ux, this.uy, this.vx, this.vy, this.w, this.h);
  final double cx, cy, ux, uy, vx, vy, w, h;
}

/// Rotating-calipers minimum-area rectangle over the convex hull of [pts].
_MinAreaRect? _minAreaRect(List<Offset> pts) {
  final hull = _convexHull(pts);
  if (hull.isEmpty) return null;
  if (hull.length == 1) {
    final p = hull[0];
    return _MinAreaRect(p.dx, p.dy, 1, 0, 0, 1, 0, 0);
  }
  if (hull.length == 2) {
    final p = hull[0], q = hull[1];
    final d = q - p;
    final len = d.distance;
    final ux = d.dx / len, uy = d.dy / len;
    return _MinAreaRect((p.dx + q.dx) / 2, (p.dy + q.dy) / 2, ux, uy, -uy, ux,
        len, 0);
  }

  double bestArea = double.infinity;
  double bcx = 0, bcy = 0, bux = 1, buy = 0, bw = 0, bh = 0;

  for (var i = 0; i < hull.length; i++) {
    final p = hull[i];
    final q = hull[(i + 1) % hull.length];
    final ex = q.dx - p.dx, ey = q.dy - p.dy;
    final elen = math.sqrt(ex * ex + ey * ey);
    if (elen < 1e-9) continue;
    final ux = ex / elen, uy = ey / elen; // edge direction
    final vxx = -uy, vyy = ux; // perpendicular

    var minU = double.infinity, maxU = double.negativeInfinity;
    var minV = double.infinity, maxV = double.negativeInfinity;
    for (final h in hull) {
      final du = h.dx * ux + h.dy * uy;
      final dv = h.dx * vxx + h.dy * vyy;
      if (du < minU) minU = du;
      if (du > maxU) maxU = du;
      if (dv < minV) minV = dv;
      if (dv > maxV) maxV = dv;
    }
    final w = maxU - minU;
    final h2 = maxV - minV;
    final areaR = w * h2;
    if (areaR < bestArea) {
      bestArea = areaR;
      final cu = (minU + maxU) / 2, cv = (minV + maxV) / 2;
      bcx = cu * ux + cv * vxx;
      bcy = cu * uy + cv * vyy;
      bux = ux;
      buy = uy;
      bw = w;
      bh = h2;
    }
  }

  // Canonicalize: u is the LONG (text) axis, pointing left-to-right; v points
  // down. Guarantees the emitted quad order tl,tr,br,bl reads upright.
  double ux = bux, uy = buy, w = bw, h = bh;
  if (w < h) {
    // Rotate axes 90 degrees: new u = old v.
    final nux = -uy, nuy = ux;
    ux = nux;
    uy = nuy;
    final t = w;
    w = h;
    h = t;
  }
  if (ux < 0 || (ux == 0 && uy < 0)) {
    ux = -ux;
    uy = -uy;
  }
  final vx = -uy, vy = ux; // for u=(1,0): v=(0,1) i.e. down in image coords

  return _MinAreaRect(bcx, bcy, ux, uy, vx, vy, w, h);
}

/// Andrew's monotone-chain convex hull (counter-clockwise in a y-down image
/// coordinate system is irrelevant here — calipers only needs the hull set).
List<Offset> _convexHull(List<Offset> pts) {
  if (pts.length <= 2) return List.of(pts);
  final sorted = List<Offset>.of(pts)
    ..sort((a, b) => a.dx != b.dx ? a.dx.compareTo(b.dx) : a.dy.compareTo(b.dy));

  double cross(Offset o, Offset a, Offset b) =>
      (a.dx - o.dx) * (b.dy - o.dy) - (a.dy - o.dy) * (b.dx - o.dx);

  final lower = <Offset>[];
  for (final p in sorted) {
    while (lower.length >= 2 &&
        cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.removeLast();
    }
    lower.add(p);
  }
  final upper = <Offset>[];
  for (final p in sorted.reversed) {
    while (upper.length >= 2 &&
        cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.removeLast();
    }
    upper.add(p);
  }
  lower.removeLast();
  upper.removeLast();
  return [...lower, ...upper];
}
