import 'dart:typed_data';

import 'package:snapseek/services/postprocessor.dart';

/// Builds an all-zero raw detector output float32[1,84,8400] (channel-major).
Float32List emptyOutput() => Float32List(kNumChannels * kNumAnchors);

/// Writes one anchor into a channel-major output buffer.
///
/// [cx],[cy],[w],[h] are in 640-letterbox space; [scores] maps COCO class id
/// (0..79) to its score. Layout: `out[c*8400 + anchor]`.
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

/// No-letterbox identity geometry: 640x640 source, scale 1, no pad. Model-space
/// pixels map 1:1 onto source pixels, so expected normalized rects are
/// pixel / 640.
PostprocessRequest identityRequest(Float32List out, {double conf = kConfThreshold}) =>
    PostprocessRequest(
      output: out,
      scale: 1.0,
      padX: 0,
      padY: 0,
      srcWidth: 640,
      srcHeight: 640,
      confThreshold: conf,
    );

/// COCO ids used across tests.
const int kClassPerson = 0;
const int kClassBackpack = 24;
const int kClassHandbag = 26;
const int kClassBottle = 39;
