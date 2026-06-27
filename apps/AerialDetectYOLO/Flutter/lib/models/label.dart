import 'dart:ui';

/// VisDrone-2019 class labels, in the exact model output order (channels 4..13).
///
/// Verified from the checkpoint `model.names` (ENOT-AutoDL/yolov8s_visdrone).
/// Order is load-bearing: the decode maps class channel index -> this list, so a
/// reorder here silently mislabels every box.
const List<String> kVisDroneLabels = <String>[
  'pedestrian', // 0
  'people', // 1
  'bicycle', // 2
  'car', // 3
  'van', // 4
  'truck', // 5
  'tricycle', // 6
  'awning-tricycle', // 7
  'bus', // 8
  'motor', // 9
];

int get kNumClasses => kVisDroneLabels.length;

/// A distinct, high-contrast color per class for the overlay.
const List<Color> kVisDroneColors = <Color>[
  Color(0xFFFF3B30), // pedestrian - red
  Color(0xFFFF9500), // people - orange
  Color(0xFFFFCC00), // bicycle - yellow
  Color(0xFF34C759), // car - green
  Color(0xFF00C7BE), // van - teal
  Color(0xFF30B0C7), // truck - cyan
  Color(0xFF007AFF), // tricycle - blue
  Color(0xFF5856D6), // awning-tricycle - indigo
  Color(0xFFAF52DE), // bus - purple
  Color(0xFFFF2D55), // motor - pink
];

Color colorForClass(int classId) =>
    kVisDroneColors[classId % kVisDroneColors.length];

String labelForClass(int classId) =>
    (classId >= 0 && classId < kVisDroneLabels.length)
    ? kVisDroneLabels[classId]
    : 'class$classId';
