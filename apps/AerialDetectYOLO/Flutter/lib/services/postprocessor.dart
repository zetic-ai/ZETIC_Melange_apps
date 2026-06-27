import 'dart:typed_data';

import '../models/detection.dart';
import 'nms.dart';
import 'preprocessor.dart';

/// Anchor count: 116^2 + 58^2 + 29^2 (928/8, /16, /32) = 17661.
const int kNumAnchors = 17661;

/// Output channels: 4 box (cx,cy,w,h) + 10 VisDrone class scores.
const int kNumChannels = 14;
const int kNumClassChannels = kNumChannels - 4; // 10

const double kDefaultConfThreshold = 0.25;
const double kDefaultIouThreshold = 0.45;

/// Decode a raw `[1,14,17661]` YOLOv8 output into final detections.
///
/// CRITICAL invariants (each guarded by a Tier A test):
///  * **Channel-major.** Stride is across the 17661 anchors, NOT across the 14
///    channels: element (channel c, anchor a) lives at `c * numAnchors + a`.
///  * **No sigmoid.** Confirmed BAKED INTO the ONNX graph (a `/model.22/Sigmoid`
///    node feeds the class branch of the output concat; class channels are
///    already probabilities in [0,1]). We must NOT apply sigmoid again here —
///    doing so would push true negatives (e.g. 0.20 -> 0.55) over threshold.
///  * **Pixel space.** Box channels are in 928 letterbox pixel space (not
///    normalized 0..1); coords are mapped back to source pixels via the inverse
///    letterbox, then clamped to the source frame.
///  * **Threshold before geometry.** The confidence gate runs before any box
///    math, so rejected anchors cost almost nothing.
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
  final int cxBase = 0;
  final int cyBase = numAnchors;
  final int wBase = 2 * numAnchors;
  final int hBase = 3 * numAnchors;
  final int classBase = 4 * numAnchors;

  final double srcW = letterbox.srcW.toDouble();
  final double srcH = letterbox.srcH.toDouble();

  for (int a = 0; a < numAnchors; a++) {
    // Max over the class channels (4..13). NO sigmoid — already probabilities.
    double bestScore = output[classBase + a];
    int bestClass = 0;
    for (int c = 1; c < numClasses; c++) {
      final double s = output[classBase + c * numAnchors + a];
      if (s > bestScore) {
        bestScore = s;
        bestClass = c;
      }
    }

    // Threshold BEFORE computing any box geometry (strict >).
    if (bestScore > confThreshold) {
      final double cx = output[cxBase + a];
      final double cy = output[cyBase + a];
      final double w = output[wBase + a];
      final double h = output[hBase + a];

      // cxcywh -> xyxy in 928 letterbox space.
      final double mx1 = cx - w / 2.0;
      final double my1 = cy - h / 2.0;
      final double mx2 = cx + w / 2.0;
      final double my2 = cy + h / 2.0;

      // Inverse letterbox into source-pixel space, then clamp to the frame.
      double x1 = letterbox.modelToSrcX(mx1).clamp(0.0, srcW);
      double y1 = letterbox.modelToSrcY(my1).clamp(0.0, srcH);
      double x2 = letterbox.modelToSrcX(mx2).clamp(0.0, srcW);
      double y2 = letterbox.modelToSrcY(my2).clamp(0.0, srcH);

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
