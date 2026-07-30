// A4 hot-path micro-benchmark (VALIDATION.md).
//
// Feeds mock tensors of the REAL shapes through the full pure-Dart hot path
// and reports per-stage medians. This is the post-processing budget — the
// only latency an agent can measure honestly — NOT end-to-end device latency
// (model.run is excluded; the served backend fixes that part).
//
// Realistic scenario per detection pass: one 1280x720 BGRA frame, a 736x736
// heatmap containing 6 text regions, and K=3 recognized crops.
//
// Run: flutter test test/benchmark/hot_path_benchmark.dart
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Rect;

import 'package:flutter_test/flutter_test.dart';
import 'package:signtranslate/config.dart';
import 'package:signtranslate/services/ctc_decoder.dart';
import 'package:signtranslate/services/db_postprocessor.dart';
import 'package:signtranslate/services/detector_preprocessor.dart';
import 'package:signtranslate/services/frame_data.dart';
import 'package:signtranslate/services/quad_deskew.dart';
import 'package:signtranslate/services/rec_preprocessor.dart';

double _median(List<double> xs) {
  final s = [...xs]..sort();
  final n = s.length;
  return n.isOdd ? s[n ~/ 2] : (s[n ~/ 2 - 1] + s[n ~/ 2]) / 2;
}

double _bench(String label, int iterations, void Function() body) {
  // Warm-up (JIT) then timed runs.
  for (var i = 0; i < 3; i++) {
    body();
  }
  final times = <double>[];
  final sw = Stopwatch();
  for (var i = 0; i < iterations; i++) {
    sw
      ..reset()
      ..start();
    body();
    sw.stop();
    times.add(sw.elapsedMicroseconds / 1000.0);
  }
  final med = _median(times);
  // ignore: avoid_print
  print('A4  ${label.padRight(34)} median ${med.toStringAsFixed(2)} ms '
      '(n=$iterations)');
  return med;
}

void main() {
  test('A4 hot-path micro-benchmark', () {
    final rng = math.Random(42);

    // --- Mock camera frame: 1280x720 BGRA with noise (realistic memory
    // access patterns; content is irrelevant).
    const fw = 1280, fh = 720;
    final bgra = Uint8List(fw * fh * 4);
    for (var i = 0; i < bgra.length; i++) {
      bgra[i] = rng.nextInt(256);
    }
    final frame = FrameData.bgra8888(
      width: fw,
      height: fh,
      bgra: bgra,
      bgraRowStride: fw * 4,
    );

    // --- Mock heatmap: 6 text regions of realistic sign sizes.
    final heatmap = Float32List(kDetInputSize * kDetInputSize);
    const regions = [
      Rect.fromLTWH(60, 180, 260, 46),
      Rect.fromLTWH(380, 190, 180, 40),
      Rect.fromLTWH(120, 300, 340, 52),
      Rect.fromLTWH(430, 420, 150, 34),
      Rect.fromLTWH(90, 500, 200, 38),
      Rect.fromLTWH(400, 560, 240, 44),
    ];
    for (final r in regions) {
      for (var y = r.top.toInt(); y < r.bottom.toInt(); y++) {
        for (var x = r.left.toInt(); x < r.right.toInt(); x++) {
          heatmap[y * kDetInputSize + x] = 0.7 + rng.nextDouble() * 0.3;
        }
      }
    }

    // --- Mock recognizer output: plausible CTC probabilities.
    final recOut = Float32List(kRecTimeSteps * kRecNumClasses);
    for (var t = 0; t < kRecTimeSteps; t++) {
      final hot = t.isEven ? rng.nextInt(kRecNumClasses) : 0;
      for (var c = 0; c < kRecNumClasses; c++) {
        recOut[t * kRecNumClasses + c] =
            c == hot ? 0.85 : 0.15 / (kRecNumClasses - 1);
      }
    }
    final decoder = CtcDecoder.fromCharsetLines(
      List.generate(836, (i) => String.fromCharCode(0x21 + (i % 90))),
    );

    // --- Pre-allocated buffers (as in production).
    final detInput = Float32List(3 * kDetInputSize * kDetInputSize);
    final recInput = Float32List(3 * kRecHeight * kRecWidth);

    // ==== Detection pass stages. ====
    late BgrFrame bgr;
    final tConvert = _bench('BGR convert (1280x720 BGRA)', 30, () {
      bgr = convertToUprightBgr(frame);
    });
    late LetterboxGeometry geo;
    final tLetterbox = _bench('detector letterbox-736 + norm', 30, () {
      geo = letterboxDetectorInput(bgr, detInput);
    });
    late DbResult db;
    final tDbPost = _bench('DB postprocess (6 regions)', 30, () {
      db = dbPostProcess(heatmap, geo);
    });
    expect(db.quads, hasLength(6));

    // ==== Per-crop stages (K = 3 crops). ====
    final crops = db.quads.sublist(0, 3);
    late BgrCrop crop;
    final tDeskew = _bench('deskew 3 quads (homography warp)', 40, () {
      for (final q in crops) {
        crop = deskewQuad(bgr, q);
      }
    });
    final tRecPre = _bench('rec preprocess 3 crops (48x320)', 40, () {
      for (var i = 0; i < 3; i++) {
        recognizerPreprocess(crop, recInput);
      }
    });
    final tCtc = _bench('CTC decode 3 outputs (40x838)', 60, () {
      for (var i = 0; i < 3; i++) {
        decoder.decode(recOut);
      }
    });

    final detStage = tConvert + tLetterbox + tDbPost;
    final recStage = tDeskew + tRecPre + tCtc;
    // ignore: avoid_print
    print('A4  ${'-' * 60}\n'
        'A4  detection-pass Dart total          ${detStage.toStringAsFixed(2)} ms\n'
        'A4  3-crop recognition Dart total      ${recStage.toStringAsFixed(2)} ms\n'
        'A4  full frame (det + 3 crops) total   '
        '${(detStage + recStage).toStringAsFixed(2)} ms');

    expect(detStage, greaterThan(0));
    expect(recStage, greaterThan(0));
  }, timeout: const Timeout(Duration(minutes: 5)));
}
