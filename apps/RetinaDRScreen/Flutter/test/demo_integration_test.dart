import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:retinadrscreen/services/postprocessor.dart';
import 'package:retinadrscreen/services/preprocessor.dart';

/// Integration harness on the validated demo images (demo_images/DEMO_IMAGES.md).
///
/// The app is UPLOAD-ONLY (no bundled sample assets), so the demo fundus files
/// are NOT app-bundled. They live in the repo `demo_images/` dir and are loaded
/// here purely as test fixtures (relative to the Flutter package root, i.e.
/// `../demo_images/`).
///
/// Real on-device ONNX inference is device-only (GATE 3), so this harness splits
/// the pipeline into the two halves an agent CAN verify without a device:
///   1. the real Dart PRE-processing path runs end-to-end on each real fundus
///      file and produces a well-formed [1,3,224,224] tensor in [-1, 1];
///   2. the POST-processing path reproduces the SPEC's measured decisions and
///      P(referable) from the exact measured logits.
void main() {
  const post = Postprocessor();

  final cases = <({
    String path,
    List<double> logits,
    double expectedP,
    bool expectedReferable,
  })>[
    (
      path: '../demo_images/IDRiD_g0_630e24b6.png',
      logits: [10.11, -0.66],
      expectedP: 0.0000,
      expectedReferable: false,
    ),
    (
      path: '../demo_images/IDRiD_g3_ca10d891.png',
      logits: [-2.72, 2.75],
      expectedP: 0.9958,
      expectedReferable: true,
    ),
    (
      path: '../demo_images/IDRiD_g4_ce3e6abe.png',
      logits: [-2.30, 2.29],
      expectedP: 0.9900,
      expectedReferable: true,
    ),
  ];

  group('preprocessing runs on the real demo fundus files', () {
    for (final c in cases) {
      test('${c.path} -> well-formed [1,3,224,224] tensor in [-1,1]', () {
        final bytes = File(c.path).readAsBytesSync();
        final data = Preprocessor.preprocess(bytes);
        expect(data.length, Preprocessor.tensorLength);
        var min = double.infinity;
        var max = double.negativeInfinity;
        for (final v in data) {
          if (v < min) min = v;
          if (v > max) max = v;
          expect(v.isFinite, isTrue);
        }
        expect(min, greaterThanOrEqualTo(-1.0 - 1e-6));
        expect(max, lessThanOrEqualTo(1.0 + 1e-6));
      });
    }
  });

  group('post-processing reproduces the measured demo decisions', () {
    for (final c in cases) {
      test('${c.path}: logits ${c.logits} -> P~${c.expectedP}', () {
        final r = post.classify(c.logits);
        expect(r.referable, c.expectedReferable);
        expect(r.pReferable, closeTo(c.expectedP, 5e-3));
      });
    }

    test('aggregate: healthy -> not-referable, diseased -> referable', () {
      final decisions =
          cases.map((c) => post.classify(c.logits).referable).toList();
      expect(decisions, [false, true, true]);
    });
  });
}
