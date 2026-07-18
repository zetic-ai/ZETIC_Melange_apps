import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:contentmoderation/models/moderation_result.dart';
import 'package:contentmoderation/services/postprocessor.dart';
import 'package:contentmoderation/services/preprocessor.dart';
import 'package:flutter_test/flutter_test.dart';

/// Integration harness on the validated demo images (demo_images/DEMO_IMAGES.md).
///
/// Real on-device ONNX inference is device-only (GATE 3), so this splits the
/// pipeline into the two halves an agent CAN verify without a device:
///   1. the real Dart PRE-processing runs end-to-end on each real demo JPEG and
///      reproduces the golden tensor within a JPEG-decode tolerance (Dart's JPEG
///      decoder differs slightly from libjpeg, so this tol is looser than the
///      lossless-PNG resize-fidelity test);
///   2. the POST-processing reproduces the measured KEEP / REVIEW / BLOCK
///      decisions and P(NSFW) from the exact onnxruntime logits (golden/meta.json).
Float32List loadGolden(String stem) {
  final bytes = File('test/fixtures/golden/$stem.f32').readAsBytesSync();
  final bd = ByteData.sublistView(bytes);
  final out = Float32List(Preprocessor.tensorLength);
  for (var i = 0; i < out.length; i++) {
    out[i] = bd.getFloat32(i * 4, Endian.little);
  }
  return out;
}

Decision decisionFromLabel(String label) => switch (label) {
      'KEEP' => Decision.keep,
      'REVIEW / BLUR' => Decision.review,
      'BLOCK' => Decision.block,
      _ => throw ArgumentError('unknown decision $label'),
    };

void main() {
  const post = Postprocessor();
  final meta = jsonDecode(
      File('test/fixtures/golden/meta.json').readAsStringSync());
  final demos = (meta['demos'] as List).cast<Map<String, dynamic>>();

  group('pre-processing reproduces the golden tensor on real demo JPEGs', () {
    for (final d in demos) {
      test('${d['file']} -> golden within JPEG-decode tolerance', () {
        final bytes =
            File('../demo_images/${d['file']}').readAsBytesSync();
        final got = Preprocessor.preprocess(bytes);
        final golden = loadGolden(d['stem'] as String);
        var maxAbs = 0.0, sum = 0.0;
        for (var i = 0; i < got.length; i++) {
          final e = (got[i] - golden[i]).abs();
          if (e > maxAbs) maxAbs = e;
          sum += e;
        }
        final meanAbs = sum / got.length;
        // ignore: avoid_print
        print('demo ${d['stem']}: maxAbs=${maxAbs.toStringAsFixed(4)} '
            'meanAbs=${meanAbs.toStringAsFixed(6)}');
        // The resampler itself is proven exact on lossless PNG in
        // resize_fidelity_test (matches golden to ~1e-6). The ONLY source of
        // divergence here is Dart-vs-libjpeg JPEG decode (chroma upsampling),
        // which is ~0.007 mean; a wrong pipeline (bilinear, bad crop, BGR) would
        // be off by >0.05 mean, so this still catches a real regression.
        expect(meanAbs, lessThan(0.015));
        // Well-formed tensor in range.
        for (final v in got) {
          expect(v, greaterThanOrEqualTo(-1.0 - 1e-6));
          expect(v, lessThanOrEqualTo(1.0 + 1e-6));
        }
      });
    }
  });

  group('post-processing reproduces the measured demo decisions', () {
    for (final d in demos) {
      test('${d['file']}: logits ${d['logits']} -> ${d['decision']}', () {
        final logits = (d['logits'] as List).cast<num>().map((e) => e.toDouble())
            .toList();
        final r = post.classify(logits);
        expect(r.pNsfw, closeTo((d['p_nsfw'] as num).toDouble(), 1e-3));
        expect(r.decision, decisionFromLabel(d['decision'] as String));
      });
    }

    test('aggregate: the demo triplet spans KEEP / REVIEW / BLOCK', () {
      final decisions = demos
          .map((d) => post
              .classify((d['logits'] as List)
                  .cast<num>()
                  .map((e) => e.toDouble())
                  .toList())
              .decision)
          .toSet();
      expect(decisions, {Decision.keep, Decision.review, Decision.block});
    });
  });
}
