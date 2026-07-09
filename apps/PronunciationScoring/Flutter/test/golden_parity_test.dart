import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sayright/models/phonemes.dart';
import 'package:sayright/services/ctc_aligner.dart';
import 'package:sayright/services/postprocessor.dart';

// Golden fixtures are produced by validation/export_golden.py running the ONNX
// (onnxruntime CPU) on the committed reference wavs. The Dart scoring head must
// reproduce the reference greedy decode / alignment / GOP from the SAME logprobs
// within 1e-3 — NOT bit-exact (on-device the served artifact differs in
// precision; here we validate the pure-Dart arithmetic against numpy).
const double kTol = 1e-3;

Map<String, dynamic> _load(String name) {
  final f = File('test/fixtures/$name');
  return jsonDecode(f.readAsStringSync()) as Map<String, dynamic>;
}

void main() {
  final index = (jsonDecode(File('test/fixtures/golden_index.json')
          .readAsStringSync()) as List)
      .cast<String>();

  test('golden index is non-empty', () {
    expect(index, isNotEmpty);
  });

  for (final clip in index) {
    group('golden parity: $clip', () {
      final fx = _load('golden_$clip.json');
      final logprobs = Float32List.fromList(
          (fx['logprobs'] as List).map((e) => (e as num).toDouble()).toList());
      final targetIds = (fx['target_ids'] as List).cast<int>();
      final lp = LogProbView(logprobs);

      test('greedy decode matches reference', () {
        final expected = (fx['greedy'] as List).cast<String>();
        expect(greedyDecode(lp).phonemes, expected);
      });

      test('blank fraction matches reference', () {
        final expected = (fx['blank_fraction'] as num).toDouble();
        expect(greedyDecode(lp).blankFraction, closeTo(expected, 1e-6));
      });

      test('forced-alignment frame sets match reference', () {
        const aligner = CtcAligner();
        final frames = aligner.align(lp, targetIds);
        final expected = (fx['aligned_frames'] as List)
            .map((e) => (e as List).cast<int>())
            .toList();
        expect(frames.length, expected.length);
        for (var i = 0; i < frames.length; i++) {
          expect(frames[i], expected[i], reason: 'phone $i frames');
        }
      });

      test('per-phoneme GOP matches reference within $kTol', () {
        const aligner = CtcAligner();
        final frames = aligner.align(lp, targetIds);
        final expected = (fx['gop'] as List).map((e) => (e as num).toDouble());
        var i = 0;
        for (final exp in expected) {
          final fr = frames[i];
          double gop = 0.0;
          if (fr.isNotEmpty) {
            var sum = 0.0;
            for (final f in fr) {
              sum += math.exp(lp.at(f, targetIds[i]));
            }
            gop = sum / fr.length;
          }
          expect(gop, closeTo(exp, kTol), reason: 'phone $i gop');
          i++;
        }
      });

      test('all target ids are phonemes (0..38)', () {
        for (final id in targetIds) {
          expect(Phonemes.isPhoneme(id), isTrue);
        }
      });
    });
  }
}
