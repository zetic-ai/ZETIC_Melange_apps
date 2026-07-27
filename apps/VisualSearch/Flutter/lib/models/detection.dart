import 'dart:ui' show Rect;

/// One detector box. [rect] is normalized 0..1 in the original (upright) frame,
/// so it maps cleanly onto any preview size. The detector is used purely as a
/// salient-object localizer, so [classId]/[label] are incidental (HUD only).
class Detection {
  const Detection({
    required this.rect,
    required this.classId,
    required this.confidence,
  });

  final Rect rect;
  final int classId;
  final double confidence;

  String get label =>
      (classId >= 0 && classId < kCocoLabels.length) ? kCocoLabels[classId] : '?';
}

/// 80 COCO class labels, in the exact channel order of YOLO11n's output
/// (`output0[4 + classId]`). Order must match Ultralytics' COCO indexing.
const List<String> kCocoLabels = <String>[
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
  'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
  'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
  'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
  'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
  'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
  'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
  'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
  'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
];
