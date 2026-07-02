import 'dart:io';
import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:retinadrgrade/models/grading_result.dart';
import 'package:retinadrgrade/services/postprocessor.dart';
import 'package:retinadrgrade/services/preprocessor.dart';

/// Integration harness on the validated demo images (../demo_images, DEMO_IMAGES.md).
///
/// GradeVue is UPLOAD-ONLY — the demo fundus images are NOT bundled as app assets.
/// This test loads them as fixtures from the repo `demo_images/` folder (one level
/// up from the Flutter package root, where `flutter test` runs).
///
/// Real on-device ONNX inference is device-only (GATE 3), so this harness splits
/// the pipeline into the two halves an agent CAN verify without a device:
///   1. the real Dart PRE-processing path runs end-to-end on each real fundus
///      file and produces a well-formed [1,3,224,224] tensor in [-1, 1];
///   2. the POST-processing path reproduces the SPEC's measured grade + 5-way
///      softmax. Since only the softmax (not raw logits) is published, we feed
///      logit_i = ln(p_i), for which softmax(logit) == p exactly, and assert the
///      argmax grade, referable flag, and full distribution reproduce DEMO_IMAGES.md.
void main() {
  const post = Postprocessor();

  // Repo demo_images path relative to the Flutter package root.
  const demoDir = '../demo_images';

  List<double> logitsFor(List<double> probs) =>
      probs.map((p) => math.log(p)).toList();

  final cases = <({
    String file,
    int grade,
    bool referable,
    List<double> softmax,
  })>[
    (
      file: 'IDRiD_g0_6389f96a.png',
      grade: 0,
      referable: false,
      softmax: [0.982, 0.008, 0.005, 0.002, 0.003],
    ),
    (
      file: 'IDRiD_g3_dd7d2789.png',
      grade: 3,
      referable: true,
      softmax: [0.007, 0.006, 0.088, 0.810, 0.089],
    ),
    (
      file: 'IDRiD_g4_278b9ee5.png',
      grade: 4,
      referable: true,
      softmax: [0.019, 0.011, 0.035, 0.125, 0.809],
    ),
  ];

  group('preprocessing runs on the real demo fundus files', () {
    for (final c in cases) {
      test('${c.file} -> well-formed [1,3,224,224] tensor in [-1,1]', () {
        final path = '$demoDir/${c.file}';
        final file = File(path);
        expect(file.existsSync(), isTrue,
            reason: 'demo fixture missing at $path');
        final bytes = file.readAsBytesSync();
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

  group('post-processing reproduces the measured demo grades', () {
    for (final c in cases) {
      test('${c.file}: -> grade ${c.grade}, referable=${c.referable}', () {
        final r = post.classify(logitsFor(c.softmax));
        expect(r.grade, c.grade);
        expect(r.referable, c.referable);
        expect(r.gradeLabel, GradingResult.gradeLabels[c.grade]);
        for (var i = 0; i < 5; i++) {
          expect(r.perGradeProbs[i], closeTo(c.softmax[i], 5e-3));
        }
      });
    }

    test('aggregate: 3/3 exact grades, referable sens/spec 1.00 on subset', () {
      final grades = cases.map((c) => post.classify(logitsFor(c.softmax)).grade);
      expect(grades, [0, 3, 4]);
      final refs =
          cases.map((c) => post.classify(logitsFor(c.softmax)).referable);
      expect(refs, [false, true, true]);
    });
  });
}
