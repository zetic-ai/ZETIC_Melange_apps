import 'dart:typed_data';
import 'dart:ui' show Rect;

import '../models/detection.dart';
import 'nms.dart';

/// Output layout constants for float32[1, 17, 8400]:
/// per anchor [cx, cy, w, h, s0..s12], CHANNEL-MAJOR — the value for channel
/// `c` of anchor `i` lives at `output[c * 8400 + i]`.
const int kNumAnchors = 8400;
const int kNumClasses = 13;
const int kNumChannels = 4 + kNumClasses; // 17
const double kIouThreshold = 0.45;

/// The lowest per-class threshold — used as the cheap pre-gate before the full
/// 13-class argmax (Tier B: threshold-before-geometry AND before full argmax).
final double kMinThreshold =
    kClassThresholds.values.reduce((a, b) => a < b ? a : b);

/// Everything one decode pass needs.
class PostprocessRequest {
  const PostprocessRequest({
    required this.output,
    required this.scale,
    required this.padX,
    required this.padY,
    required this.srcWidth,
    required this.srcHeight,
  });

  /// Raw model output, flattened float32[1, 17, 8400] (channel-major). Class
  /// scores are ALREADY sigmoid-applied by the ONNX head — no activation here.
  final Float32List output;
  final double scale;
  final int padX;
  final int padY;
  final int srcWidth;
  final int srcHeight;
}

/// Decodes the raw tensor into final detections:
///
/// 1. Cheap pre-gate: read only the 4 whitelisted class scores; if all are
///    below the smallest per-class threshold, skip the anchor (typical case).
/// 2. Full argmax over ALL 13 classes — semantics must match the Stage-0
///    harness: if a non-whitelisted class (e.g. Person) outscores the
///    whitelisted ones, the anchor is DROPPED, not relabeled.
/// 3. Per-class threshold on the winning score (Hardhat 0.25, others 0.15).
///    Scores are used as-is: the sigmoid is baked into the ONNX. Applying a
///    second sigmoid would silently crush every confidence toward 0.5-0.73.
/// 4. Box geometry only for survivors: cxcywh -> corners (640 letterbox
///    space) -> inverse letterbox -> normalized 0..1 of the upright frame.
/// 5. PER-CLASS NMS, IoU 0.45.
List<Detection> postprocessOutput(PostprocessRequest req) {
  final out = req.output;
  assert(out.length == kNumChannels * kNumAnchors,
      'expected ${kNumChannels * kNumAnchors} floats, got ${out.length}');
  const int n = kNumAnchors;

  final candidates = <Detection>[];
  final double invSrcW = 1.0 / req.srcWidth;
  final double invSrcH = 1.0 / req.srcHeight;
  final double invScale = 1.0 / req.scale;
  final double minGate = kMinThreshold;

  // Channel base offsets (score channel for class c is at (4 + c) * n).
  const int hardhatOff = (4 + kClassHardhat) * n;
  const int noHardhatOff = (4 + kClassNoHardhat) * n;
  const int noVestOff = (4 + kClassNoVest) * n;
  const int vestOff = (4 + kClassVest) * n;

  for (var i = 0; i < n; i++) {
    // 1. Pre-gate on the four rendered classes only.
    final double sHardhat = out[hardhatOff + i];
    final double sNoHardhat = out[noHardhatOff + i];
    final double sNoVest = out[noVestOff + i];
    final double sVest = out[vestOff + i];
    double best = sHardhat;
    if (sNoHardhat > best) best = sNoHardhat;
    if (sNoVest > best) best = sNoVest;
    if (sVest > best) best = sVest;
    if (best < minGate) continue;

    // 2. Full argmax over all 13 classes (whitelist-vs-rest semantics).
    var bestClass = 0;
    var bestScore = out[4 * n + i];
    for (var c = 1; c < kNumClasses; c++) {
      final double s = out[(4 + c) * n + i];
      if (s > bestScore) {
        bestScore = s;
        bestClass = c;
      }
    }
    final double? threshold = kClassThresholds[bestClass];
    if (threshold == null) continue; // winner not whitelisted -> drop anchor
    if (bestScore < threshold) continue;

    // 4. Geometry for survivors only.
    final double cx = out[i];
    final double cy = out[n + i];
    final double w = out[2 * n + i];
    final double h = out[3 * n + i];

    double x1 = (cx - w / 2 - req.padX) * invScale;
    double y1 = (cy - h / 2 - req.padY) * invScale;
    double x2 = (cx + w / 2 - req.padX) * invScale;
    double y2 = (cy + h / 2 - req.padY) * invScale;

    final double nx1 = (x1 * invSrcW).clamp(0.0, 1.0);
    final double ny1 = (y1 * invSrcH).clamp(0.0, 1.0);
    final double nx2 = (x2 * invSrcW).clamp(0.0, 1.0);
    final double ny2 = (y2 * invSrcH).clamp(0.0, 1.0);
    if (nx2 <= nx1 || ny2 <= ny1) continue;

    candidates.add(Detection(
      rect: Rect.fromLTRB(nx1, ny1, nx2, ny2),
      classId: bestClass,
      confidence: bestScore,
    ));
  }

  return nmsPerClass(candidates, kIouThreshold);
}

/// Expected flat length of a valid output buffer (sanity checks / tests).
int get expectedOutputLength => kNumChannels * kNumAnchors;
