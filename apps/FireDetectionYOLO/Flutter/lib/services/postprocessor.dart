import 'dart:typed_data';
import 'dart:ui' show Rect;

import '../models/detection.dart';
import 'nms.dart';
import 'preprocessor.dart' show kInputSize;

/// Output layout constants for float32[1, 6, 8400].
const int kNumAnchors = 8400;
const int kNumChannels = 6; // cx, cy, w, h, fire_conf, smoke_conf
const double kIouThreshold = 0.45;

/// Bundles everything [postprocessOutput] needs so it can run in a [compute]
/// isolate.
class PostprocessRequest {
  const PostprocessRequest({
    required this.output,
    required this.confThreshold,
    required this.scale,
    required this.padX,
    required this.padY,
    required this.srcWidth,
    required this.srcHeight,
  });

  /// Raw model output, flattened float32[1, 6, 8400] (channel-major).
  final Float32List output;
  final double confThreshold;
  final double scale;
  final int padX;
  final int padY;
  final int srcWidth;
  final int srcHeight;
}

/// Top-level so it can run in a [compute] isolate.
///
/// The tensor is channel-major: value for channel `c`, anchor `i` lives at
/// `output[c * 8400 + i]`. We pull the per-anchor box + class scores, threshold,
/// convert center form to corners, undo the letterbox back into original-frame
/// pixels, normalize to 0..1, then run per-class NMS.
List<Detection> postprocessOutput(PostprocessRequest req) {
  final out = req.output;
  const int n = kNumAnchors;
  final candidates = <Detection>[];

  final double invSrcW = 1.0 / req.srcWidth;
  final double invSrcH = 1.0 / req.srcHeight;

  for (var i = 0; i < n; i++) {
    final double fire = out[4 * n + i];
    final double smoke = out[5 * n + i];

    final double conf;
    final int classId;
    if (fire >= smoke) {
      conf = fire;
      classId = kClassFire;
    } else {
      conf = smoke;
      classId = kClassSmoke;
    }
    if (conf <= req.confThreshold) continue;

    final double cx = out[i];
    final double cy = out[n + i];
    final double w = out[2 * n + i];
    final double h = out[3 * n + i];

    // Center form -> corners, still in 640 letterboxed space.
    double x1 = cx - w / 2;
    double y1 = cy - h / 2;
    double x2 = cx + w / 2;
    double y2 = cy + h / 2;

    // Undo letterbox -> original-frame pixels.
    x1 = (x1 - req.padX) / req.scale;
    y1 = (y1 - req.padY) / req.scale;
    x2 = (x2 - req.padX) / req.scale;
    y2 = (y2 - req.padY) / req.scale;

    // Normalize to 0..1 of the source frame and clamp to bounds.
    final double nx1 = (x1 * invSrcW).clamp(0.0, 1.0);
    final double ny1 = (y1 * invSrcH).clamp(0.0, 1.0);
    final double nx2 = (x2 * invSrcW).clamp(0.0, 1.0);
    final double ny2 = (y2 * invSrcH).clamp(0.0, 1.0);
    if (nx2 <= nx1 || ny2 <= ny1) continue;

    candidates.add(Detection(
      rect: Rect.fromLTRB(nx1, ny1, nx2, ny2),
      classId: classId,
      confidence: conf,
    ));
  }

  return nmsPerClass(candidates, kIouThreshold, classCount: kClassLabels.length);
}

/// Sanity helper: expected flat length of a valid output buffer.
int get expectedOutputLength => kNumChannels * kNumAnchors;

/// Re-exported so callers don't import preprocessor just for the size constant.
int get modelInputSize => kInputSize;
