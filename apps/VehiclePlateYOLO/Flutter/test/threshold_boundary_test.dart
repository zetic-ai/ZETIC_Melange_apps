import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vehicleplateyolo/services/letterbox.dart';
import 'package:vehicleplateyolo/services/postprocessor.dart';

/// TRAP: off-by-a-hair threshold handling. Spec uses STRICT '>' at 0.25, so an
/// anchor exactly at 0.25 is dropped and 0.2501 is kept.
void main() {
  test('strict > 0.25: 0.2499 and 0.2500 dropped, 0.2501 kept', () {
    const n = 8400;
    final out = Float32List(5 * n);

    // Three well-separated boxes so NMS never collapses survivors.
    void plant(int a, double cx, double cy, double conf) {
      out[0 * n + a] = cx;
      out[1 * n + a] = cy;
      out[2 * n + a] = 20;
      out[3 * n + a] = 20;
      out[4 * n + a] = conf;
    }

    plant(0, 50, 50, 0.2499); // below
    plant(1, 300, 300, 0.2500); // exactly at threshold -> dropped (strict >)
    plant(2, 600, 600, 0.2501); // above -> kept

    final params = LetterboxParams.forImage(640, 640, 640);
    const post = Postprocessor(); // default confThreshold 0.25
    final dets = post.decode(out, params);

    expect(dets.length, 1);
    expect(dets.single.confidence, closeTo(0.2501, 1e-6));
  });
}
