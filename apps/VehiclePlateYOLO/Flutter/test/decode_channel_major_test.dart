import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vehicleplateyolo/services/letterbox.dart';
import 'package:vehicleplateyolo/services/postprocessor.dart';

/// TRAP: [1,5,8400] is CHANNEL-major. Channel c, anchor a lives at c*8400+a
/// (stride across the 8400 anchors), NOT at a*5+c. A row-major reader produces
/// plausible-but-wrong boxes that are impossible to spot in a live demo.
void main() {
  test('decodes channel-major stride (c*8400+a), not row-major', () {
    const n = 8400;
    final out = Float32List(5 * n);

    // Plant ONE plate at anchor a=100, writing each field at its channel base.
    const a = 100;
    out[0 * n + a] = 320; // cx
    out[1 * n + a] = 320; // cy
    out[2 * n + a] = 64; // w
    out[3 * n + a] = 32; // h
    out[4 * n + a] = 0.9; // plate_conf (sigmoid already baked in)

    // Identity letterbox (src == model 640) so model space == image space.
    final params = LetterboxParams.forImage(640, 640, 640);
    const post = Postprocessor();
    final dets = post.decode(out, params);

    expect(dets.length, 1, reason: 'a row-major read would find nothing here');
    final d = dets.first;
    expect(d.left, closeTo(288, 1e-3)); // 320 - 64/2
    expect(d.right, closeTo(352, 1e-3));
    expect(d.top, closeTo(304, 1e-3)); // 320 - 32/2
    expect(d.bottom, closeTo(336, 1e-3));
    expect(d.confidence, closeTo(0.9, 1e-6));
  });
}
