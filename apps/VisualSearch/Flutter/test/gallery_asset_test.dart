import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:snapseek/services/gallery.dart';

/// Loads the REAL bundled gallery (vectors computed by the exported
/// visualsearch_embed.onnx) and asserts the model contract that on-device
/// cosine relies on: every catalog vector is L2-normalized (‖v‖≈1, proving the
/// in-graph L2-norm), so a plain dot product == cosine. Also checks
/// self-retrieval ranks the identical item first.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('bundled gallery loads, is non-trivial, and every vector is unit-norm',
      () async {
    final gallery = await Gallery.load();
    expect(gallery.dim, 512);
    expect(gallery.items.length, greaterThanOrEqualTo(40));

    for (final it in gallery.items) {
      expect(it.vector.length, 512);
      double n = 0;
      for (final x in it.vector) {
        n += x * x;
      }
      expect(math.sqrt(n), closeTo(1.0, 1e-3),
          reason: '${it.id} not unit-norm');
    }
  });

  test('self-retrieval: a catalog vector ranks itself #1 with cosine ~1',
      () async {
    final gallery = await Gallery.load();
    final probe = gallery.items[3];
    final res = gallery.topK(probe.vector, k: 3);
    expect(res.first.item.id, probe.id);
    expect(res.first.score, closeTo(1.0, 1e-3));
  });
}
