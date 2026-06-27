import 'dart:typed_data';

import '../models/detection.dart';
import 'letterbox.dart';
import 'nms.dart';

/// Decodes the raw model output into license-plate detections.
///
/// Output is float32 **[1,5,8400], channel-major**: channel `c`, anchor `a`
/// lives at index `c*8400 + a` (stride across the 8400 anchors, NOT across the
/// 5). Per anchor: [cx, cy, w, h, plate_conf] in 640x640 letterboxed pixel
/// space. Confidence already has Sigmoid baked into the exported graph, so NO
/// extra activation is applied here (the silent score-semantics trap).
class Postprocessor {
  const Postprocessor({
    this.numAnchors = 8400,
    this.confThreshold = 0.25,
    this.iouThreshold = 0.45,
  });

  final int numAnchors;
  final double confThreshold;
  final double iouThreshold;

  List<Detection> decode(Float32List output, LetterboxParams params) {
    final n = numAnchors;
    // Channel base offsets (channel-major).
    final cxBase = 0 * n;
    final cyBase = 1 * n;
    final wBase = 2 * n;
    final hBase = 3 * n;
    final confBase = 4 * n;

    final candidates = <Detection>[];
    for (var a = 0; a < n; a++) {
      // Threshold BEFORE geometry so rejected anchors are nearly free.
      // Strict '>' per spec; sigmoid already baked in -> use as-is.
      final conf = output[confBase + a];
      if (conf <= confThreshold) continue;

      final cx = output[cxBase + a];
      final cy = output[cyBase + a];
      final w = output[wBase + a];
      final h = output[hBase + a];

      // cxcywh -> xyxy in 640 space.
      final halfW = w / 2.0;
      final halfH = h / 2.0;
      // Undo letterbox into upright source-image space.
      final left = params.inverseX(cx - halfW);
      final top = params.inverseY(cy - halfH);
      final right = params.inverseX(cx + halfW);
      final bottom = params.inverseY(cy + halfH);

      candidates.add(
        Detection(
          left: left,
          top: top,
          right: right,
          bottom: bottom,
          confidence: conf,
        ),
      );
    }

    return nonMaxSuppression(candidates, iouThreshold: iouThreshold);
  }
}
