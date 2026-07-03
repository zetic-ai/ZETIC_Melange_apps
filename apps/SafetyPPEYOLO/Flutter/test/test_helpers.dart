import 'dart:typed_data';

import 'package:siteguard/services/postprocessor.dart';

/// Builds an all-zero raw output tensor float32[1,17,8400] (channel-major).
Float32List emptyOutput() => Float32List(kNumChannels * kNumAnchors);

/// Writes one anchor into a channel-major output buffer.
///
/// [cx], [cy], [w], [h] are in 640-letterbox space; [scores] maps class id
/// (0..12) to its sigmoid-space score.
void setAnchor(
  Float32List out,
  int anchor, {
  required double cx,
  required double cy,
  required double w,
  required double h,
  required Map<int, double> scores,
}) {
  const int n = kNumAnchors;
  out[anchor] = cx;
  out[n + anchor] = cy;
  out[2 * n + anchor] = w;
  out[3 * n + anchor] = h;
  scores.forEach((classId, score) {
    out[(4 + classId) * n + anchor] = score;
  });
}

/// A no-letterbox identity geometry: 640x640 source, scale 1, no padding.
/// With it, model-space pixels map 1:1 onto source pixels, so expected
/// normalized rects are just pixel / 640.
PostprocessRequest identityRequest(Float32List out) => PostprocessRequest(
      output: out,
      scale: 1.0,
      padX: 0,
      padY: 0,
      srcWidth: 640,
      srcHeight: 640,
    );
