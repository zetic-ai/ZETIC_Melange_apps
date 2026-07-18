import 'dart:typed_data';
import 'dart:ui' show Rect;

import '../models/detection.dart';
import 'nms.dart';

/// Output layout for float32[1,84,8400], channel-major.
const int kNumAnchors = 8400;
const int kNumClasses = 80;
const int kNumChannels = 4 + kNumClasses; // cx,cy,w,h + 80 class scores = 84

/// Defaults from the spec.
const double kConfThreshold = 0.25;
const double kIouThreshold = 0.5;

/// Bundles everything [postprocessDetect] needs so it can run in a [compute]
/// isolate. Geometry is the letterbox forward transform to invert.
class PostprocessRequest {
  const PostprocessRequest({
    required this.output,
    required this.scale,
    required this.padX,
    required this.padY,
    required this.srcWidth,
    required this.srcHeight,
    this.confThreshold = kConfThreshold,
  });

  /// Raw model output, flattened float32[1,84,8400], channel-major.
  final Float32List output;
  final double scale;
  final int padX;
  final int padY;
  final int srcWidth;
  final int srcHeight;
  final double confThreshold;
}

/// Expected flat length of a valid detector output buffer.
int get expectedOutputLength => kNumChannels * kNumAnchors;

/// Decode YOLO11n [1,84,8400] (channel-major: value for channel `c`, anchor `i`
/// is `output[c*8400 + i]` — stride across anchors, NOT across the 84), apply
/// the confidence threshold BEFORE box geometry, convert cxcywh→xyxy, invert
/// the letterbox back to original-frame pixels, normalize to 0..1, then run
/// GLOBAL (class-agnostic) NMS. `result.first` is the primary box.
///
/// Top-level so it can run in a [compute] isolate.
List<Detection> postprocessDetect(PostprocessRequest req) {
  assert(req.output.length == expectedOutputLength,
      'detector output must be $expectedOutputLength floats (1x84x8400)');

  final out = req.output;
  const int n = kNumAnchors;
  final candidates = <Detection>[];

  final double invSrcW = 1.0 / req.srcWidth;
  final double invSrcH = 1.0 / req.srcHeight;

  for (var i = 0; i < n; i++) {
    // Class-agnostic salient score = max over the 80 class channels.
    double best = out[4 * n + i];
    int bestClass = 0;
    for (var c = 1; c < kNumClasses; c++) {
      final double s = out[(4 + c) * n + i];
      if (s > best) {
        best = s;
        bestClass = c;
      }
    }
    if (best <= req.confThreshold) continue; // threshold BEFORE geometry

    final double cx = out[i];
    final double cy = out[n + i];
    final double w = out[2 * n + i];
    final double h = out[3 * n + i];

    // Center form → corners, still in 640 letterbox space.
    double x1 = cx - w / 2;
    double y1 = cy - h / 2;
    double x2 = cx + w / 2;
    double y2 = cy + h / 2;

    // Undo letterbox → original-frame pixels (exact reverse: subtract pad,
    // divide by scale).
    x1 = (x1 - req.padX) / req.scale;
    y1 = (y1 - req.padY) / req.scale;
    x2 = (x2 - req.padX) / req.scale;
    y2 = (y2 - req.padY) / req.scale;

    // Normalize to 0..1 of the source frame and clamp.
    final double nx1 = (x1 * invSrcW).clamp(0.0, 1.0);
    final double ny1 = (y1 * invSrcH).clamp(0.0, 1.0);
    final double nx2 = (x2 * invSrcW).clamp(0.0, 1.0);
    final double ny2 = (y2 * invSrcH).clamp(0.0, 1.0);
    if (nx2 <= nx1 || ny2 <= ny1) continue;

    candidates.add(Detection(
      rect: Rect.fromLTRB(nx1, ny1, nx2, ny2),
      classId: bestClass,
      confidence: best,
    ));
  }

  return globalNms(candidates, kIouThreshold);
}

/// Pick the primary salient box: highest confidence, ties broken by larger
/// area. Returns null when the detector found nothing (→ center-crop fallback).
Detection? primaryBox(List<Detection> dets) {
  if (dets.isEmpty) return null;
  Detection best = dets.first;
  for (final d in dets) {
    if (d.confidence > best.confidence ||
        (d.confidence == best.confidence &&
            d.rect.width * d.rect.height >
                best.rect.width * best.rect.height)) {
      best = d;
    }
  }
  return best;
}
