import 'dart:ui';

/// Dental pathology class labels, in the EXACT model output order (class
/// channels 4..6 of the `[1,7,8400]` output).
///
/// Verified from the checkpoint `model.names` (liodon-ai/dental-panoramic-
/// detector). Order is load-bearing: the decode maps class-channel index -> this
/// list, so a reorder here silently mislabels every box.
const List<String> kDentalLabels = <String>[
  'caries', // 0
  'periapical_lesion', // 1
  'impacted_tooth', // 2
];

int get kNumClasses => kDentalLabels.length;

/// A distinct, high-contrast color per class for the overlay.
const List<Color> kDentalColors = <Color>[
  Color(0xFFFF453A), // caries - red
  Color(0xFFFFD60A), // periapical_lesion - amber
  Color(0xFF32D74B), // impacted_tooth - green
];

Color colorForClass(int classId) =>
    kDentalColors[classId % kDentalColors.length];

String labelForClass(int classId) =>
    (classId >= 0 && classId < kDentalLabels.length)
    ? kDentalLabels[classId]
    : 'class$classId';

/// Human-friendly label for chips/UI (underscores -> spaces).
String prettyLabel(int classId) => labelForClass(classId).replaceAll('_', ' ');
