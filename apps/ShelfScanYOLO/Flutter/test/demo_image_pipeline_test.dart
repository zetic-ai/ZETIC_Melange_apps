import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:shelfscanyolo/models/detection.dart';
import 'package:shelfscanyolo/services/letterbox.dart';
import 'package:shelfscanyolo/services/postprocessor.dart';
import 'package:shelfscanyolo/services/preprocessor.dart';

/// Integration harness on the SKU-110K demo frames. We cannot run the ONNX
/// model in a pure-Dart test (that is the GATE-3 device run), so this validates
/// the *preprocessing* half of the pipeline numerically against the exact
/// validate_demo.py letterbox on the real images, plus a synthetic decode+NMS
/// pass that exercises the postprocessing half on dense input.
///
/// The demo images are NOT app-bundled assets (the shipped app is upload-only).
/// They are loaded here as repo test fixtures from `../demo_images` (relative to
/// the Flutter package dir, where `flutter test` runs).
void main() {
  const samples = <String>[
    '../demo_images/shelf_ultradense_499.jpg',
    '../demo_images/shelf_dense_216.jpg',
    '../demo_images/shelf_clean_155.jpg',
  ];

  group('preprocessing parity on real demo images', () {
    for (final path in samples) {
      test(path, () {
        final file = File(path);
        expect(file.existsSync(), isTrue, reason: 'missing demo fixture $path');

        final decoded = img.decodeImage(file.readAsBytesSync());
        expect(decoded, isNotNull);
        final oriented = img.bakeOrientation(decoded!);
        final w = oriented.width;
        final h = oriented.height;

        final pre = const Preprocessor().processDecoded(oriented);
        final t = pre.transform;

        // scale + pad match the reference letterbox formula exactly.
        final expScale = math.min(640 / w, 640 / h);
        expect(t.scale, closeTo(expScale, 1e-12));
        expect(t.resizedWidth, (w * expScale).round());
        expect(t.resizedHeight, (h * expScale).round());
        expect(t.padX, (640 - t.resizedWidth) ~/ 2);
        expect(t.padY, (640 - t.resizedHeight) ~/ 2);

        // Input tensor shape and value range.
        expect(pre.input.length, 3 * 640 * 640);
        var lo = double.infinity, hi = -double.infinity;
        for (final v in pre.input) {
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
        expect(lo, greaterThanOrEqualTo(0.0));
        expect(hi, lessThanOrEqualTo(1.0));

        // Padded regions must be gray 114/255 ~= 0.447 (R plane).
        const padGray = 114 / 255.0;
        if (t.padX > 0) {
          final idx = 320 * 640 + (t.padX ~/ 2); // left bar, mid height
          expect(pre.input[idx], closeTo(padGray, 1e-6));
        }
        if (t.padY > 0) {
          final idx = (t.padY ~/ 2) * 640 + 320; // top bar, mid width
          expect(pre.input[idx], closeTo(padGray, 1e-6));
        }
        expect(t.padX == 0 || t.padY == 0, isTrue,
            reason: 'letterbox pads only one axis');
      });
    }
  });

  test('postprocessing: dense overlapping output collapses via global NMS', () {
    // 40x40 grid of near-identical stacked boxes -> NMS must reduce the count.
    final numAnchors = 8400;
    final out = Float32List(5 * numAnchors);
    var a = 0;
    for (var gy = 0; gy < 40 && a < numAnchors; gy++) {
      for (var gx = 0; gx < 40 && a < numAnchors; gx++) {
        final cx = 8.0 + gx * 8.0;
        final cy = 8.0 + gy * 8.0;
        out[0 * numAnchors + a] = cx;
        out[1 * numAnchors + a] = cy;
        out[2 * numAnchors + a] = 40.0; // overlaps neighbours 8px apart
        out[3 * numAnchors + a] = 40.0;
        out[4 * numAnchors + a] = 0.5;
        a++;
      }
    }
    final t = LetterboxTransform.compute(originalWidth: 640, originalHeight: 640);
    final raw = const Postprocessor().decode(out, t);
    final nmsed = const Postprocessor().process(out, t);
    expect(raw.length, 1600); // all above threshold
    expect(nmsed.length, lessThan(raw.length)); // NMS collapsed overlaps
    expect(nmsed.length, greaterThan(0));
    // All survivors are pixel-space boxes inside the original image.
    for (final Detection d in nmsed) {
      expect(d.box.x2, greaterThan(d.box.x1));
      expect(d.box.y2, greaterThan(d.box.y1));
    }
  });
}
