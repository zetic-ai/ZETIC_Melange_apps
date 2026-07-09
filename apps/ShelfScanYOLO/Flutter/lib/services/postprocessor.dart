import 'dart:typed_data';

import '../models/detection.dart';
import 'letterbox.dart';
import 'nms.dart';

/// Decodes the raw YOLO11s output `float32[1,5,8400]` into product detections
/// in original-image pixel space, then applies single-class global NMS.
///
/// CRITICAL correctness facts for THIS model (see SPEC):
///  - Output is **channel-major**: for anchor `a`, the 5 values live at strides
///    `[0*8400+a, 1*8400+a, 2*8400+a, 3*8400+a, 4*8400+a]` = [cx,cy,w,h,score].
///    Stride across the 8400 anchors, NEVER read 5 contiguous floats.
///  - The class score is **already sigmoid-activated in-graph** — use it AS-IS.
///    Do NOT apply sigmoid again.
///  - Box coords are in **640 letterbox pixel space** (not normalized 0..1).
///  - Threshold is **strict `> confThreshold`** (exclusive), applied BEFORE any
///    box geometry, to match validate_demo.py.
class Postprocessor {
  const Postprocessor({
    this.confThreshold = defaultConfThreshold,
    this.iouThreshold = defaultIouThreshold,
    this.numAnchors = 8400,
  });

  static const double defaultConfThreshold = 0.25;
  static const double defaultIouThreshold = 0.45;

  final double confThreshold;
  final double iouThreshold;
  final int numAnchors;

  /// Full pipeline: decode -> global NMS.
  List<Detection> process(Float32List output, LetterboxTransform transform) {
    return nonMaxSuppression(decode(output, transform), iouThreshold);
  }

  /// Decode + threshold + cxcywh->xyxy + inverse-letterbox (no NMS).
  List<Detection> decode(Float32List output, LetterboxTransform transform) {
    final expected = 5 * numAnchors;
    if (output.length != expected) {
      throw ArgumentError(
        'Expected $expected floats ([1,5,$numAnchors]), got ${output.length}.',
      );
    }
    // Channel base offsets (channel-major layout).
    const cxBase = 0;
    final cyBase = numAnchors;
    final wBase = 2 * numAnchors;
    final hBase = 3 * numAnchors;
    final scoreBase = 4 * numAnchors;

    final detections = <Detection>[];
    for (var a = 0; a < numAnchors; a++) {
      final score = output[scoreBase + a]; // already sigmoid'd -> use as-is
      if (score > confThreshold) {
        // Threshold BEFORE decoding geometry -> rejected anchors cost ~nothing.
        final cx = output[cxBase + a];
        final cy = output[cyBase + a];
        final w = output[wBase + a];
        final h = output[hBase + a];
        // cxcywh -> xyxy in 640 letterbox px.
        final halfW = w / 2;
        final halfH = h / 2;
        final lx1 = cx - halfW;
        final ly1 = cy - halfH;
        final lx2 = cx + halfW;
        final ly2 = cy + halfH;
        // Invert letterbox -> original-image px.
        detections.add(
          Detection(
            BBox(
              transform.toOriginalX(lx1),
              transform.toOriginalY(ly1),
              transform.toOriginalX(lx2),
              transform.toOriginalY(ly2),
            ),
            score,
          ),
        );
      }
    }
    return detections;
  }
}
