import 'dart:math' as math;
import 'dart:ui' show Offset, Rect;

/// An ordered text-region quadrilateral: corners are top-left, top-right,
/// bottom-right, bottom-left. Coordinate space is documented at each use site
/// (736×736 letterboxed model space vs upright frame space vs screen space) —
/// a [Quad] itself is space-agnostic.
class Quad {
  Quad(this.tl, this.tr, this.br, this.bl);

  /// Builds an axis-aligned quad from a rectangle (tests + synthetic data).
  Quad.fromRect(Rect r)
      : tl = r.topLeft,
        tr = r.topRight,
        br = r.bottomRight,
        bl = r.bottomLeft;

  /// Orders four arbitrary corners into tl/tr/br/bl.
  ///
  /// Robust for any rotation (including exactly 45°, where the classic
  /// min(x+y)/min(y−x) rule ties and can return DUPLICATE corners): the
  /// points are sorted clockwise by angle around their centroid (screen
  /// coordinates, y down), then rotated so the corner with the smallest
  /// x+y (tie-broken by y, then x) leads as top-left.
  factory Quad.ordered(List<Offset> pts) {
    assert(pts.length == 4, 'Quad needs exactly 4 points');
    final cx = (pts[0].dx + pts[1].dx + pts[2].dx + pts[3].dx) / 4;
    final cy = (pts[0].dy + pts[1].dy + pts[2].dy + pts[3].dy) / 4;
    final clockwise = [...pts]..sort((a, b) => math
        .atan2(a.dy - cy, a.dx - cx)
        .compareTo(math.atan2(b.dy - cy, b.dx - cx)));

    var start = 0;
    for (var i = 1; i < 4; i++) {
      final p = clockwise[i], s = clockwise[start];
      final dSum = (p.dx + p.dy) - (s.dx + s.dy);
      if (dSum < -1e-9 ||
          (dSum.abs() <= 1e-9 &&
              (p.dy < s.dy || (p.dy == s.dy && p.dx < s.dx)))) {
        start = i;
      }
    }
    return Quad(
      clockwise[start],
      clockwise[(start + 1) % 4],
      clockwise[(start + 2) % 4],
      clockwise[(start + 3) % 4],
    );
  }

  final Offset tl;
  final Offset tr;
  final Offset br;
  final Offset bl;

  List<Offset> get points => [tl, tr, br, bl];

  Rect get boundingBox {
    var minX = tl.dx, maxX = tl.dx, minY = tl.dy, maxY = tl.dy;
    for (final p in [tr, br, bl]) {
      minX = math.min(minX, p.dx);
      maxX = math.max(maxX, p.dx);
      minY = math.min(minY, p.dy);
      maxY = math.max(maxY, p.dy);
    }
    return Rect.fromLTRB(minX, minY, maxX, maxY);
  }

  Offset get center =>
      Offset((tl.dx + tr.dx + br.dx + bl.dx) / 4, (tl.dy + tr.dy + br.dy + bl.dy) / 4);

  /// Shoelace polygon area.
  double get area {
    final p = points;
    var sum = 0.0;
    for (var i = 0; i < 4; i++) {
      final a = p[i];
      final b = p[(i + 1) % 4];
      sum += a.dx * b.dy - b.dx * a.dy;
    }
    return sum.abs() / 2;
  }

  /// Applies [f] to every corner.
  Quad map(Offset Function(Offset) f) => Quad(f(tl), f(tr), f(br), f(bl));

  /// Axis-aligned bounding-box IoU against [other] — the RegionTracker cache
  /// key metric (cheap, and sufficient for "same sign, camera moved a bit").
  double bboxIou(Quad other) {
    final a = boundingBox;
    final b = other.boundingBox;
    final ix = math.max(
      0.0,
      math.min(a.right, b.right) - math.max(a.left, b.left),
    );
    final iy = math.max(
      0.0,
      math.min(a.bottom, b.bottom) - math.max(a.top, b.top),
    );
    final inter = ix * iy;
    final union = a.width * a.height + b.width * b.height - inter;
    if (union <= 0) return 0;
    return inter / union;
  }

  @override
  String toString() => 'Quad($tl, $tr, $br, $bl)';
}

/// One recognized region as shown by the overlay: the quad in upright frame
/// space plus the decoded string and its CTC confidence.
class RecognizedRegion {
  const RecognizedRegion({
    required this.quad,
    required this.text,
    required this.confidence,
    required this.fromCache,
  });

  final Quad quad;

  /// Empty string means "region detected but decoded to nothing" — the
  /// overlay draws the outline without a label chip.
  final String text;
  final double confidence;

  /// True when the string was served by the IoU cache (no recognizer run).
  final bool fromCache;
}
