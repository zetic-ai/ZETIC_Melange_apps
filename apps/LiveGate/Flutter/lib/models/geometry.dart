/// Geometry primitives shared by the PAD and FACE branches.
///
/// All coordinates are in the **upright** image space — the space ML Kit reports
/// face boxes and landmarks in, and the space [FrameConverter.toUprightBgr]
/// produces. Keeping one coordinate space end-to-end is what makes the crop and
/// the warp reconcile with the detector (the on-device reconciliation risk noted
/// in HANDOFF.md is exactly about keeping these spaces in lockstep).
library;

/// A face bounding box in upright-image pixels.
class FaceBox {
  const FaceBox({
    required this.x,
    required this.y,
    required this.w,
    required this.h,
  });

  final double x;
  final double y;
  final double w;
  final double h;

  double get cx => x + w / 2;
  double get cy => y + h / 2;
}

/// The five ArcFace alignment landmarks, in upright-image pixels, in the exact
/// order the ArcFace template expects: left eye, right eye, nose tip, left mouth
/// corner, right mouth corner.
class Landmarks5 {
  const Landmarks5({
    required this.leftEye,
    required this.rightEye,
    required this.nose,
    required this.mouthLeft,
    required this.mouthRight,
  });

  final Point2 leftEye;
  final Point2 rightEye;
  final Point2 nose;
  final Point2 mouthLeft;
  final Point2 mouthRight;

  /// The five points in ArcFace template order.
  List<Point2> get ordered =>
      [leftEye, rightEye, nose, mouthLeft, mouthRight];
}

/// A simple double-precision 2-D point (Dart's [Point] is fine but this keeps
/// the pipeline free of `dart:math` boxing surprises and reads clearly).
class Point2 {
  const Point2(this.x, this.y);
  final double x;
  final double y;
}

/// The PAD 2.7x margin crop box, in upright-image pixels. Corners are inclusive
/// of the source pixels that get sampled ([x2]/[y2] are the far edge, matching
/// the Python `frame[y1:y2+1, x1:x2+1]` slice in the spec recipe).
class CropBox {
  const CropBox(this.x1, this.y1, this.x2, this.y2);
  final double x1;
  final double y1;
  final double x2;
  final double y2;

  double get width => x2 - x1;
  double get height => y2 - y1;
}
