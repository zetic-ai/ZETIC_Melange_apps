import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:livedocredact/services/ctc_decoder.dart';
import 'package:livedocredact/services/db_postprocessor.dart';
import 'package:livedocredact/services/detector_preprocessor.dart';
import 'package:livedocredact/services/frame_data.dart';
import 'package:livedocredact/services/pii_classifier.dart';
import 'package:livedocredact/services/recognizer_preprocessor.dart';
import 'package:livedocredact/services/region_tracker.dart';

/// A4 hot-path micro-benchmark (VALIDATION.md).
///
/// Feeds mock tensors of the REAL shapes through the full pure-Dart hot path
/// at a realistic region count (N=10 detected fields, K=3 recognizer budget)
/// and reports the median per stage. This is the post-processing budget — the
/// only latency number producible without a device — and the Tier-B baseline.
/// Model time (det ~129 ms + K x ~32 ms on the CPU fallback) is additive and
/// fixed by Melange.
void main() {
  const int iterations = 60;
  const int nBlobs = 10;
  const int budgetK = kRecognizerBudgetK;

  double median(List<double> xs) {
    final s = List<double>.of(xs)..sort();
    final n = s.length;
    return n.isOdd ? s[n ~/ 2] : (s[n ~/ 2 - 1] + s[n ~/ 2]) / 2;
  }

  test('A4 hot-path micro-benchmark (mock tensors, N=$nBlobs, K=$budgetK)',
      () {
    final rnd = math.Random(42);

    // --- Mock 720p BGRA frame with document-like light background + dark
    // text bands (content matters little for timing; branches stay real).
    const w = 1280, h = 720;
    final bgra = Uint8List(w * h * 4);
    for (var i = 0; i < w * h; i++) {
      final v = (i ~/ w) % 60 < 8 ? 40 : 235; // dark line every 60 rows
      bgra[i * 4] = v;
      bgra[i * 4 + 1] = v;
      bgra[i * 4 + 2] = v;
      bgra[i * 4 + 3] = 255;
    }
    final frame = UprightFrame(FrameData.bgra8888(
        width: w, height: h, bgra: bgra, bgraRowStride: w * 4));

    // --- Mock heatmap: nBlobs text fields in 640-space.
    final heatmap = Float32List(kDetInputSize * kDetInputSize);
    final blobRects = <Rect>[];
    for (var i = 0; i < nBlobs; i++) {
      final x0 = 40 + (i % 2) * 320;
      final y0 = 150 + (i ~/ 2) * 65;
      const bw = 220, bh = 26;
      blobRects.add(Rect.fromLTWH(
          x0.toDouble(), y0.toDouble(), bw.toDouble(), bh.toDouble()));
      for (var y = y0; y < y0 + bh; y++) {
        for (var x = x0; x < x0 + bw; x++) {
          heatmap[y * kDetInputSize + x] = 0.85 + rnd.nextDouble() * 0.1;
        }
      }
    }

    // --- Mock recognizer output: plausible peaked distributions.
    final logits = Float32List(kCtcSteps * kCtcClasses);
    for (var t = 0; t < kCtcSteps; t++) {
      for (var c = 0; c < kCtcClasses; c++) {
        logits[t * kCtcClasses + c] = rnd.nextDouble() * 0.002;
      }
      logits[t * kCtcClasses + (t % 3 == 0 ? 0 : 1 + rnd.nextInt(436))] =
          0.85;
    }
    final dictChars =
        List<String>.generate(436, (i) => String.fromCharCode(33 + (i % 94)));
    final decoder = CtcDecoder(dictChars);
    final classifier = PiiClassifier();

    final detBuf = Float32List(3 * kDetInputSize * kDetInputSize);
    final recBuf = Float32List(kRecTensorLength);

    final tPre = <double>[];
    final tDecode = <double>[];
    final tTrack = <double>[];
    final tCrop = <double>[];
    final tCtc = <double>[];
    final tPii = <double>[];
    final tTotal = <double>[];

    late LetterboxGeometry geometry;
    final sw = Stopwatch();

    // Untimed JIT warm-up so the timed medians measure steady-state code.
    for (var it = 0; it < 10; it++) {
      final det = preprocessDetectorFrame(frame, out: detBuf);
      final db = decodeDbHeatmap(heatmap, det.geometry);
      final tracker = RegionTracker()..update(db.regions);
      for (final r in tracker.scheduleRecognition(budgetK)) {
        preprocessRecognizerCrop(frame, r.quad, out: recBuf);
        decoder.decode(logits);
      }
    }

    for (var it = 0; it < iterations; it++) {
      final frameSw = Stopwatch()..start();

      sw
        ..reset()
        ..start();
      final det = preprocessDetectorFrame(frame, out: detBuf);
      geometry = det.geometry;
      tPre.add(sw.elapsedMicroseconds / 1000.0);

      sw
        ..reset()
        ..start();
      final db = decodeDbHeatmap(heatmap, geometry);
      tDecode.add(sw.elapsedMicroseconds / 1000.0);
      expect(db.regions.length, nBlobs,
          reason: 'benchmark must exercise the real region count');

      // Fresh tracker per iteration so scheduling cost stays realistic
      // (every iteration schedules a full K batch).
      sw
        ..reset()
        ..start();
      final tracker = RegionTracker();
      tracker.update(db.regions);
      final scheduled = tracker.scheduleRecognition(budgetK);
      tTrack.add(sw.elapsedMicroseconds / 1000.0);
      expect(scheduled.length, budgetK);

      sw
        ..reset()
        ..start();
      for (final r in scheduled) {
        preprocessRecognizerCrop(frame, r.quad, out: recBuf);
      }
      tCrop.add(sw.elapsedMicroseconds / 1000.0);

      sw
        ..reset()
        ..start();
      for (var k = 0; k < budgetK; k++) {
        decoder.decode(logits);
      }
      tCtc.add(sw.elapsedMicroseconds / 1000.0);

      sw
        ..reset()
        ..start();
      final inputs = [
        for (var i = 0; i < db.regions.length; i++)
          PiiInputField(
            bbox: db.regions[i].bbox,
            text: i % 3 == 0
                ? 'DOB 12/05/1988'
                : (i % 3 == 1 ? 'JOHN Q PUBLIC' : 'P<UTODOE<<JOHN<<<<<<<<'),
            confidence: 0.9,
          ),
      ];
      classifier.classify(inputs);
      tPii.add(sw.elapsedMicroseconds / 1000.0);

      tTotal.add(frameSw.elapsedMicroseconds / 1000.0);
    }

    String fmt(List<double> xs) => median(xs).toStringAsFixed(2);
    // Printed report — this is the A4 deliverable (medians over $iterations
    // iterations) and the Tier-B baseline.
    // ignore: avoid_print
    print('--- A4 hot-path micro-benchmark '
        '(median of $iterations iters, N=$nBlobs regions, K=$budgetK) ---');
    // ignore: avoid_print
    print('det preprocess (letterbox+norm+NCHW 640x640): ${fmt(tPre)} ms');
    // ignore: avoid_print
    print('DB decode (binarize+CC+calipers+unclip+inv):  ${fmt(tDecode)} ms');
    // ignore: avoid_print
    print('tracker update + budget schedule:             ${fmt(tTrack)} ms');
    // ignore: avoid_print
    print('$budgetK x recognizer crop/warp/pad [1,3,48,320]:    '
        '${fmt(tCrop)} ms');
    // ignore: avoid_print
    print('$budgetK x CTC greedy decode [1,40,438]:             '
        '${fmt(tCtc)} ms');
    // ignore: avoid_print
    print('PII classify ($nBlobs fields):                     ${fmt(tPii)} ms');
    // ignore: avoid_print
    print('TOTAL pure-Dart hot path per frame:           ${fmt(tTotal)} ms');

    // Sanity floor: the whole Dart hot path must stay well under one CPU
    // detector pass (~129 ms) so Dart is never the bottleneck.
    expect(median(tTotal), lessThan(129),
        reason: 'pure-Dart hot path must not dominate the CPU-fallback '
            'model time');
  }, timeout: const Timeout(Duration(minutes: 5)));
}
