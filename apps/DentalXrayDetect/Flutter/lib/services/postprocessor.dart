import 'dart:typed_data';

import '../models/detection.dart';
import 'nms.dart';
import 'preprocessor.dart';

/// Anchor count: 80^2 + 40^2 + 20^2 (640/8, /16, /32) = 8400.
const int kNumAnchors = 8400;

/// Output channels: 4 box (cx,cy,w,h) + 3 dental class scores = 7.
const int kNumChannels = 7;
const int kNumClassChannels = kNumChannels - 4; // 3

/// Confidence gate. **0.45** is the validated operating point (model card +
/// measured on the DENTEX demo set). At 0.25 caries over-fires on adjacent
/// healthy teeth (precision collapses to ~0.33) — do NOT lower the default.
const double kDefaultConfThreshold = 0.45;

/// Per-class NMS IoU. **0.45** per SPEC + the demo validation harness (NOT the
/// model card's 0.35 inference setting).
const double kDefaultIouThreshold = 0.45;

/// Decode a raw `[1,7,8400]` YOLO11n output into final detections.
///
/// CRITICAL invariants (each guarded by a Tier A test):
///  * **Channel-major.** Stride is across the 8400 anchors, NOT across the 7
///    channels: element (channel c, anchor a) lives at `c * numAnchors + a`.
///    Reading it row-major (7 contiguous per anchor) is the classic decode bug
///    and produces garbage boxes.
///  * **No sigmoid.** The 3 class scores are ALREADY sigmoid-activated in the
///    ONNX graph (verified in [0,1] via onnxruntime). We must NOT re-apply
///    sigmoid — a second activation compresses the dynamic range toward
///    ~0.5–0.62 and silently breaks the 0.45 threshold.
///  * **Pixel space.** Box channels are in 640 letterbox PIXEL space (not
///    normalized 0..1); coords are mapped back to original-image pixels via the
///    inverse letterbox, then clamped to the image frame.
///  * **Threshold before geometry (strict `>`).** The confidence gate runs
///    before any box math, so rejected anchors cost almost nothing. Strict `>`
///    matches the validation harness (`conf > conf_thres`) so the Dart pipeline
///    reproduces the measured demo detections exactly. (SPEC's prose says
///    "≥ 0.45"; the harness is the source of truth for reproduction and uses
///    strict `>`, so a score of exactly 0.45 is dropped.)
///  * **Per-class NMS.** Different-class overlaps both survive.
List<Detection> decodeDetections(
  Float32List output,
  LetterboxParams letterbox, {
  int numClasses = kNumClassChannels,
  int numAnchors = kNumAnchors,
  double confThreshold = kDefaultConfThreshold,
  double iouThreshold = kDefaultIouThreshold,
}) {
  final List<Detection> candidates = <Detection>[];

  // Precompute channel base offsets (channel c starts at c * numAnchors).
  const int cxBase = 0;
  final int cyBase = numAnchors;
  final int wBase = 2 * numAnchors;
  final int hBase = 3 * numAnchors;
  final int classBase = 4 * numAnchors;

  final double srcW = letterbox.srcW.toDouble();
  final double srcH = letterbox.srcH.toDouble();

  for (int a = 0; a < numAnchors; a++) {
    // Max over the 3 class channels (4..6). NO sigmoid — already probabilities.
    double bestScore = output[classBase + a];
    int bestClass = 0;
    for (int c = 1; c < numClasses; c++) {
      final double s = output[classBase + c * numAnchors + a];
      if (s > bestScore) {
        bestScore = s;
        bestClass = c;
      }
    }

    // Threshold BEFORE computing any box geometry (strict `>`, harness-exact).
    if (bestScore > confThreshold) {
      final double cx = output[cxBase + a];
      final double cy = output[cyBase + a];
      final double w = output[wBase + a];
      final double h = output[hBase + a];

      // cxcywh -> xyxy in 640 letterbox space.
      final double mx1 = cx - w / 2.0;
      final double my1 = cy - h / 2.0;
      final double mx2 = cx + w / 2.0;
      final double my2 = cy + h / 2.0;

      // Inverse letterbox into original-image pixel space, then clamp.
      final double x1 = letterbox.modelToSrcX(mx1).clamp(0.0, srcW);
      final double y1 = letterbox.modelToSrcY(my1).clamp(0.0, srcH);
      final double x2 = letterbox.modelToSrcX(mx2).clamp(0.0, srcW);
      final double y2 = letterbox.modelToSrcY(my2).clamp(0.0, srcH);

      candidates.add(Detection(
        left: x1,
        top: y1,
        right: x2,
        bottom: y2,
        classId: bestClass,
        confidence: bestScore,
      ));
    }
  }

  return nonMaxSuppressionPerClass(candidates, iouThreshold);
}
