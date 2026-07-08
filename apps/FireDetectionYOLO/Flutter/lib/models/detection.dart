import 'dart:ui' show Rect;

/// Class indices as emitted by the YOLO11s fire/smoke model.
const int kClassFire = 0;
const int kClassSmoke = 1;
const List<String> kClassLabels = ['fire', 'smoke'];

/// A single post-processed detection.
///
/// [rect] is normalized to 0..1 in the **original camera frame** space
/// (letterboxing already undone), so the overlay can map it onto the preview
/// regardless of the model's 640x640 input size.
class Detection {
  const Detection({
    required this.rect,
    required this.classId,
    required this.confidence,
  });

  final Rect rect;
  final int classId;
  final double confidence;

  String get label => kClassLabels[classId];

  bool get isFire => classId == kClassFire;
  bool get isSmoke => classId == kClassSmoke;
}
