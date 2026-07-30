import 'dart:ui';

import 'label.dart';

/// One detected object.
///
/// Box coordinates are stored as plain doubles in **source-image pixel space**
/// (the camera buffer's coordinate system, after the letterbox inverse) — NOT
/// the 928 model space and NOT normalized 0..1. The overlay maps these to the
/// screen with the same BoxFit/orientation transform used for the preview.
///
/// All fields are primitives so the object copies cleanly across the isolate
/// boundary (no `dart:ui` objects are stored; [rect] is computed lazily).
class Detection {
  const Detection({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
    required this.classId,
    required this.confidence,
  });

  final double left;
  final double top;
  final double right;
  final double bottom;
  final int classId;
  final double confidence;

  double get width => right - left;
  double get height => bottom - top;
  double get area => width * height;

  String get label => labelForClass(classId);

  Rect get rect => Rect.fromLTRB(left, top, right, bottom);

  @override
  String toString() =>
      'Detection($label ${(confidence * 100).toStringAsFixed(0)}% '
      '[${left.toStringAsFixed(0)},${top.toStringAsFixed(0)},'
      '${right.toStringAsFixed(0)},${bottom.toStringAsFixed(0)}])';
}
