import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Offset;

import '../config.dart';
import '../models/text_region.dart';
import 'detector_preprocessor.dart';

/// Result of DB post-processing: text-region quads in upright FRAME space
/// (letterbox already inverted), in reading order, plus the raw heatmap
/// stats surfaced on the HUD (the dashboard accuracy-row anomaly check:
/// a served detector returning a degenerate map shows up here immediately).
class DbResult {
  const DbResult({
    required this.quads,
    required this.mapMin,
    required this.mapMax,
    required this.mapMean,
  });

  final List<Quad> quads;
  final double mapMin;
  final double mapMax;
  final double mapMean;
}

/// DBPostProcess in pure Dart over the detector's [1,1,736,736] probability
/// map (Sigmoid is BAKED into the ONNX — no activation is applied here):
///
/// 1. binarize at [probThreshold] (0.3)
/// 2. 8-connected components (iterative BFS — no recursion)
/// 3. drop components whose MEAN probability < [boxThreshold] (0.6)
/// 4. min-area rotated rect (convex hull + rotating calipers)
/// 5. unclip by [unclipRatio] (1.5): offset d = area·ratio / perimeter
/// 6. exact letterbox inverse into frame space
/// 7. reading order: top→bottom bands, then left→right
DbResult dbPostProcess(
  Float32List probMap,
  LetterboxGeometry geometry, {
  double probThreshold = kDbProbThreshold,
  double boxThreshold = kDbBoxThreshold,
  double unclipRatio = kDbUnclipRatio,
}) {
  const size = kDetInputSize;
  const area = size * size;
  assert(probMap.length >= area, 'probability map must be 736*736');

  // --- Heatmap stats (single pass) + binarization. -------------------------
  var mapMin = double.infinity;
  var mapMax = -double.infinity;
  var mapSum = 0.0;
  // 0 = background, 1 = unvisited text, 2 = visited.
  final bin = Uint8List(area);
  for (var i = 0; i < area; i++) {
    final v = probMap[i];
    if (v < mapMin) mapMin = v;
    if (v > mapMax) mapMax = v;
    mapSum += v;
    if (v > probThreshold) bin[i] = 1;
  }

  // --- Connected components. ------------------------------------------------
  final quads = <Quad>[];
  final stack = <int>[];
  final boundary = <Offset>[];

  for (var start = 0; start < area; start++) {
    if (bin[start] != 1) continue;

    var count = 0;
    var probSum = 0.0;
    boundary.clear();
    stack.add(start);
    bin[start] = 2;

    while (stack.isNotEmpty) {
      final idx = stack.removeLast();
      final x = idx % size;
      final y = idx ~/ size;
      count++;
      probSum += probMap[idx];

      var isBoundary = false;
      for (var dy = -1; dy <= 1; dy++) {
        final ny = y + dy;
        if (ny < 0 || ny >= size) {
          isBoundary = true;
          continue;
        }
        for (var dx = -1; dx <= 1; dx++) {
          if (dx == 0 && dy == 0) continue;
          final nx = x + dx;
          if (nx < 0 || nx >= size) {
            isBoundary = true;
            continue;
          }
          final nIdx = ny * size + nx;
          final state = bin[nIdx];
          if (state == 0) {
            isBoundary = true;
          } else if (state == 1) {
            bin[nIdx] = 2;
            stack.add(nIdx);
          }
        }
      }
      if (isBoundary) boundary.add(Offset(x.toDouble(), y.toDouble()));
    }

    // Mean-probability (box_thresh) filter.
    if (count == 0 || probSum / count < boxThreshold) continue;
    if (boundary.length < 3) continue;

    // Min-area rotated rect over the component boundary.
    final rect = _minAreaRect(_convexHull(boundary));
    if (rect == null) continue;
    if (math.min(rect.w, rect.h) < kDbMinBoxSize) continue;

    // Unclip: DB shrinks text kernels during training; dilate the fitted box
    // by d = area·ratio/perimeter (uniform rect expansion — equivalent to the
    // polygon offset for rectangles).
    final rectArea = rect.w * rect.h;
    final perimeter = 2 * (rect.w + rect.h);
    final d = perimeter > 0 ? rectArea * unclipRatio / perimeter : 0.0;
    final w = rect.w + 2 * d;
    final h = rect.h + 2 * d;

    // Corners in 736-space, clamped to the map, then letterbox-inverted.
    final hw = w / 2, hh = h / 2;
    final corners = <Offset>[
      Offset(rect.cx - rect.ux * hw - rect.vx * hh,
          rect.cy - rect.uy * hw - rect.vy * hh),
      Offset(rect.cx + rect.ux * hw - rect.vx * hh,
          rect.cy + rect.uy * hw - rect.vy * hh),
      Offset(rect.cx + rect.ux * hw + rect.vx * hh,
          rect.cy + rect.uy * hw + rect.vy * hh),
      Offset(rect.cx - rect.ux * hw + rect.vx * hh,
          rect.cy - rect.uy * hw + rect.vy * hh),
    ]
        .map((p) => Offset(
            p.dx.clamp(0.0, size.toDouble()), p.dy.clamp(0.0, size.toDouble())))
        .map(geometry.toFrame)
        .toList();

    quads.add(Quad.ordered(corners));
  }

  return DbResult(
    quads: sortReadingOrder(quads),
    mapMin: mapMin.isFinite ? mapMin : 0,
    mapMax: mapMax.isFinite ? mapMax : 0,
    mapMean: mapSum / area,
  );
}

/// Sorts quads into reading order: rows (bands) top→bottom, then left→right
/// within a band. Band breaks occur where the vertical gap between adjacent
/// y-centers exceeds 60% of the median quad height.
List<Quad> sortReadingOrder(List<Quad> quads) {
  if (quads.length < 2) return quads;

  final heights = quads.map((q) => q.boundingBox.height).toList()..sort();
  final medianHeight = heights[heights.length ~/ 2];
  final tolerance = math.max(1.0, medianHeight * 0.6);

  final byY = [...quads]..sort((a, b) => a.center.dy.compareTo(b.center.dy));
  final bands = <List<Quad>>[];
  for (final q in byY) {
    if (bands.isEmpty ||
        q.center.dy - bands.last.last.center.dy > tolerance) {
      bands.add([q]);
    } else {
      bands.last.add(q);
    }
  }
  return [
    for (final band in bands)
      ...band..sort((a, b) => a.center.dx.compareTo(b.center.dx)),
  ];
}

// ---------------------------------------------------------------------------
// Geometry helpers.
// ---------------------------------------------------------------------------

/// Andrew's monotone-chain convex hull.
List<Offset> _convexHull(List<Offset> points) {
  final pts = [...points]..sort(
      (a, b) => a.dx != b.dx ? a.dx.compareTo(b.dx) : a.dy.compareTo(b.dy));
  if (pts.length <= 2) return pts;

  double cross(Offset o, Offset a, Offset b) =>
      (a.dx - o.dx) * (b.dy - o.dy) - (a.dy - o.dy) * (b.dx - o.dx);

  final lower = <Offset>[];
  for (final p in pts) {
    while (lower.length >= 2 &&
        cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.removeLast();
    }
    lower.add(p);
  }
  final upper = <Offset>[];
  for (final p in pts.reversed) {
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

class _RotatedRect {
  const _RotatedRect(this.cx, this.cy, this.w, this.h, this.ux, this.uy)
      : vx = -uy,
        vy = ux;

  final double cx, cy; // center
  final double w, h; // extents along (ux,uy) / (vx,vy)
  final double ux, uy; // unit vector of the w axis
  final double vx, vy; // unit vector of the h axis (perpendicular)
}

/// Rotating calipers: the minimum-area rectangle over a convex hull has one
/// side collinear with a hull edge.
_RotatedRect? _minAreaRect(List<Offset> hull) {
  if (hull.length < 3) return null;

  _RotatedRect? best;
  var bestArea = double.infinity;

  for (var i = 0; i < hull.length; i++) {
    final a = hull[i];
    final b = hull[(i + 1) % hull.length];
    final ex = b.dx - a.dx;
    final ey = b.dy - a.dy;
    final len = math.sqrt(ex * ex + ey * ey);
    if (len < 1e-9) continue;
    final ux = ex / len, uy = ey / len;
    final vx = -uy, vy = ux;

    var minU = double.infinity, maxU = -double.infinity;
    var minV = double.infinity, maxV = -double.infinity;
    for (final p in hull) {
      final u = p.dx * ux + p.dy * uy;
      final v = p.dx * vx + p.dy * vy;
      if (u < minU) minU = u;
      if (u > maxU) maxU = u;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    final w = maxU - minU;
    final h = maxV - minV;
    final area = w * h;
    if (area < bestArea) {
      bestArea = area;
      final cu = (minU + maxU) / 2;
      final cv = (minV + maxV) / 2;
      best = _RotatedRect(
        cu * ux + cv * vx,
        cu * uy + cv * vy,
        w,
        h,
        ux,
        uy,
      );
    }
  }
  return best;
}
