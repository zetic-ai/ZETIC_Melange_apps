import 'dart:ui' show Rect;

/// Class ids exactly as emitted by ajayshah/SafetyPPEYOLO
/// (ayushgupta7777/safetyvision-yolov8 v2, 13 classes).
///
/// Only the four ids in [kRenderedClassIds] are ever emitted by the
/// postprocessor. Person (11) is DEGENERATE in this checkpoint — measured 0
/// predictions on the Stage-0 ground-truth set even at conf 0.05 — and must
/// never be rendered or counted (SPEC.md). Mask/NO-Mask were excluded by the
/// GATE-2 ruling (no toggle).
const int kClassHardhat = 3;
const int kClassNoHardhat = 7;
const int kClassNoVest = 9;
const int kClassPerson = 11; // never rendered — degenerate, see SPEC.md
const int kClassVest = 12;

/// All 13 model class names, in model id order (0-12).
const List<String> kModelClassNames = [
  'Fall-Detected',
  'Gloves',
  'Goggles',
  'Hardhat',
  'Mask',
  'NO-Gloves',
  'NO-Goggles',
  'NO-Hardhat',
  'NO-Mask',
  'NO-Safety Vest',
  'No_Harness',
  'Person',
  'Safety Vest',
];

/// The GATE-2-approved render whitelist: exactly {3, 7, 9, 12}.
const Set<int> kRenderedClassIds = {
  kClassHardhat,
  kClassNoHardhat,
  kClassNoVest,
  kClassVest,
};

/// Per-class confidence thresholds (SPEC.md, measured at Stage 0):
/// Hardhat 0.25; Safety Vest and the violation classes 0.15 (vest recall
/// 0.26 -> 0.35 at precision 0.94 when lowered).
const Map<int, double> kClassThresholds = {
  kClassHardhat: 0.25,
  kClassNoHardhat: 0.15,
  kClassNoVest: 0.15,
  kClassVest: 0.15,
};

/// Short labels for the HUD/overlay chips.
const Map<int, String> kDisplayLabels = {
  kClassHardhat: 'HARDHAT',
  kClassNoHardhat: 'NO HARDHAT',
  kClassNoVest: 'NO VEST',
  kClassVest: 'VEST',
};

/// A single post-processed detection.
///
/// [rect] is normalized to 0..1 in the **original upright camera frame**
/// (letterboxing already undone), so the overlay can map it onto the preview
/// regardless of the model's 640x640 input size.
class Detection {
  const Detection({
    required this.rect,
    required this.classId,
    required this.confidence,
  });

  final Rect rect;

  /// Original model class id (one of [kRenderedClassIds]).
  final int classId;

  /// Sigmoid-applied score straight from the model output (0..1). The model
  /// bakes the sigmoid into the ONNX head; the postprocessor must NOT apply a
  /// second one.
  final double confidence;

  String get label => kDisplayLabels[classId] ?? kModelClassNames[classId];

  /// True for the "missing PPE" classes (rendered red).
  bool get isViolation =>
      classId == kClassNoHardhat || classId == kClassNoVest;
}
