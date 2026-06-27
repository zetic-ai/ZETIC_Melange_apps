import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vehicleplateyolo/services/letterbox.dart';
import 'package:vehicleplateyolo/services/postprocessor.dart';

/// TRAP: outputs are in 640x640 PIXEL space (cx,cy,w,h), not normalized 0-1.
/// If the decoder assumed normalized and multiplied by 640, every box would be
/// 640x too big and off-screen.
void main() {
  test('treats cx,cy,w,h as 640px pixels, not normalized 0-1', () {
    const n = 8400;
    final out = Float32List(5 * n);
    const a = 4200;
    out[0 * n + a] = 320; // center of a 640px image
    out[1 * n + a] = 160;
    out[2 * n + a] = 100;
    out[3 * n + a] = 50;
    out[4 * n + a] = 0.8;

    final params = LetterboxParams.forImage(640, 640, 640); // identity
    const post = Postprocessor();
    final d = post.decode(out, params).single;

    final cx = (d.left + d.right) / 2;
    final cy = (d.top + d.bottom) / 2;
    expect(cx, closeTo(320, 1e-3)); // used directly, NOT 320*640
    expect(cy, closeTo(160, 1e-3));
    expect(d.width, closeTo(100, 1e-3));
    expect(d.height, closeTo(50, 1e-3));
  });
}
