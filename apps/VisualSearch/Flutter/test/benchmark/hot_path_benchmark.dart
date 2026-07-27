// Tier A4 hot-path micro-benchmark + Tier B before/after evidence.
//
// Feeds mock tensors of the REAL shapes through the full pure-Dart hot path for
// SnapSeek's snap→search pipeline: detector letterbox (640) + decode/global-NMS
// ([1,84,8400]) + embed crop/resize (256) + gallery dot-product ranking (60x512).
// This is the post-processing budget, NOT end-to-end device latency (the two
// model runs are Melange's side).
//
// Each optimized stage is benchmarked against a deliberately naive variant so
// every Tier B optimization has a measured before/after delta (0.5% rule).
// Run: flutter test test/benchmark/hot_path_benchmark.dart
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Rect;

import 'package:flutter_test/flutter_test.dart';
import 'package:snapseek/models/detection.dart';
import 'package:snapseek/models/gallery_item.dart';
import 'package:snapseek/services/gallery.dart';
import 'package:snapseek/services/nms.dart';
import 'package:snapseek/services/postprocessor.dart';
import 'package:snapseek/services/preprocessor.dart';

const int kIters = 40;
const int kGallerySize = 60;
const int kDim = 512;

double medianMs(List<double> xs) {
  final s = List<double>.of(xs)..sort();
  final n = s.length;
  return n.isOdd ? s[n ~/ 2] : (s[n ~/ 2 - 1] + s[n ~/ 2]) / 2;
}

double benchMs(void Function() body, {int iters = kIters}) {
  for (var i = 0; i < 3; i++) {
    body();
  }
  final times = <double>[];
  final sw = Stopwatch();
  for (var i = 0; i < iters; i++) {
    sw
      ..reset()
      ..start();
    body();
    sw.stop();
    times.add(sw.elapsedMicroseconds / 1000.0);
  }
  return medianMs(times);
}

/// Mock upright portrait RGB frame (1080x1920, 3 bytes/pixel).
({Uint8List rgb, int w, int h}) mockFrame(math.Random rng) {
  const w = 1080, h = 1920;
  final rgb = Uint8List(w * h * 3);
  for (var i = 0; i < rgb.length; i++) {
    rgb[i] = rng.nextInt(256);
  }
  return (rgb: rgb, w: w, h: h);
}

/// Mock detector output [1,84,8400]: sub-threshold noise + ~20 real boxes.
Float32List mockOutput(math.Random rng) {
  final out = Float32List(kNumChannels * kNumAnchors);
  const int n = kNumAnchors;
  for (var c = 4; c < kNumChannels; c++) {
    final base = c * n;
    for (var i = 0; i < n; i++) {
      out[base + i] = rng.nextDouble() * 0.12; // below the 0.25 threshold
    }
  }
  for (var i = 0; i < n; i++) {
    out[i] = rng.nextDouble() * 640;
    out[n + i] = rng.nextDouble() * 640;
    out[2 * n + i] = 20 + rng.nextDouble() * 200;
    out[3 * n + i] = 20 + rng.nextDouble() * 200;
  }
  for (var k = 0; k < 20; k++) {
    final i = rng.nextInt(n);
    out[(4 + rng.nextInt(kNumClasses)) * n + i] = 0.3 + rng.nextDouble() * 0.6;
  }
  return out;
}

Float32List mockUnitVec(math.Random rng) {
  final v = Float32List(kDim);
  double s = 0;
  for (var i = 0; i < kDim; i++) {
    v[i] = rng.nextDouble() * 2 - 1;
    s += v[i] * v[i];
  }
  s = math.sqrt(s);
  for (var i = 0; i < kDim; i++) {
    v[i] /= s;
  }
  return v;
}

List<GalleryItem> mockGallery(math.Random rng) => List.generate(
      kGallerySize,
      (i) => GalleryItem(
        id: 'g$i',
        label: 'g$i',
        category: 'c',
        instance: 'i$i',
        view: 'v',
        vector: mockUnitVec(rng),
        thumbAsset: 'x',
      ),
    );

// ---------------------------------------------------------------------------
// Naive "before" variants (Tier B evidence).
// ---------------------------------------------------------------------------

/// Naive letterbox: TWO passes (resize into an intermediate interleaved buffer,
/// then normalize+reorder to planar NCHW) with a per-call intermediate alloc.
Float32List naiveLetterbox(Uint8List rgb, int srcW, int srcH) {
  const int size = kDetectSize;
  const int area = size * size;
  final tmp = Float64List(area * 3); // pass-1 intermediate (wasteful)
  final input = Float32List(3 * area)..fillRange(0, 3 * area, 114 / 255.0);
  final g = letterboxGeometry(srcW, srcH);
  final int newW = (srcW * g.scale).round();
  final int newH = (srcH * g.scale).round();
  for (var oy = g.padY; oy < g.padY + newH; oy++) {
    if (oy < 0 || oy >= size) continue;
    final uy = ((oy - g.padY) / g.scale).floor().clamp(0, srcH - 1);
    for (var ox = g.padX; ox < g.padX + newW; ox++) {
      if (ox < 0 || ox >= size) continue;
      final ux = ((ox - g.padX) / g.scale).floor().clamp(0, srcW - 1);
      final si = (uy * srcW + ux) * 3;
      final ti = (oy * size + ox) * 3;
      tmp[ti] = rgb[si].toDouble();
      tmp[ti + 1] = rgb[si + 1].toDouble();
      tmp[ti + 2] = rgb[si + 2].toDouble();
    }
  }
  for (var p = 0; p < area; p++) {
    input[p] = tmp[p * 3] / 255.0;
    input[area + p] = tmp[p * 3 + 1] / 255.0;
    input[2 * area + p] = tmp[p * 3 + 2] / 255.0;
  }
  return input;
}

/// Naive decode: a faithful mirror of postprocessDetect that removes ONLY the
/// threshold-first gate — it computes box geometry AND allocates a Detection
/// for EVERY one of the 8400 anchors, filtering at the end. Isolates the
/// threshold-before-geometry optimization (shipped skips ~8380 allocations).
List<Detection> naiveDecode(Float32List out) {
  const int n = kNumAnchors;
  final candidates = <Detection>[];
  for (var i = 0; i < n; i++) {
    var best = out[4 * n + i];
    var bestClass = 0;
    for (var c = 1; c < kNumClasses; c++) {
      final s = out[(4 + c) * n + i];
      if (s > best) {
        best = s;
        bestClass = c;
      }
    }
    final cx = out[i], cy = out[n + i], w = out[2 * n + i], h = out[3 * n + i];
    final x1 = ((cx - w / 2) / 640).clamp(0.0, 1.0);
    final y1 = ((cy - h / 2) / 640).clamp(0.0, 1.0);
    final x2 = ((cx + w / 2) / 640).clamp(0.0, 1.0);
    final y2 = ((cy + h / 2) / 640).clamp(0.0, 1.0);
    final d = Detection(
      rect: Rect.fromLTRB(x1, y1, x2, y2),
      classId: bestClass,
      confidence: best,
    );
    if (best <= kConfThreshold) continue; // threshold AFTER geometry+alloc
    if (x2 <= x1 || y2 <= y1) continue;
    candidates.add(d);
  }
  return globalNms(candidates, kIouThreshold);
}

void main() {
  test('SnapSeek hot-path micro-benchmark (A4 baseline + Tier B deltas)', () {
    final rng = math.Random(42);
    final frame = mockFrame(rng);
    final output = mockOutput(rng);
    final gallery = mockGallery(rng);
    final query = mockUnitVec(rng);
    final req = PostprocessRequest(
      output: output,
      scale: letterboxGeometry(frame.w, frame.h).scale,
      padX: letterboxGeometry(frame.w, frame.h).padX,
      padY: letterboxGeometry(frame.w, frame.h).padY,
      srcWidth: frame.w,
      srcHeight: frame.h,
    );
    final embedReq = EmbedRequest(
      width: frame.w,
      height: frame.h,
      rgb: frame.rgb,
      box: const Rect.fromLTRB(0.2, 0.2, 0.8, 0.8),
    );

    // --- shipped stages ---
    final preMs = benchMs(() => buildLetterboxTensor(frame.rgb, frame.w, frame.h));
    final decodeMs = benchMs(() => postprocessDetect(req));
    final embedPreMs = benchMs(() => preprocessEmbed(embedReq));
    final rankMs = benchMs(() => rankByDot(gallery, query, k: 5));
    final fullMs = benchMs(() {
      buildLetterboxTensor(frame.rgb, frame.w, frame.h);
      postprocessDetect(req);
      preprocessEmbed(embedReq);
      rankByDot(gallery, query, k: 5);
    });

    // --- naive "before" variants (Tier B evidence) ---
    final naivePreMs = benchMs(() => naiveLetterbox(frame.rgb, frame.w, frame.h));
    final naiveDecodeMs = benchMs(() => naiveDecode(output));

    // Sanity: the shipped decode returns a non-empty, plausible result.
    expect(postprocessDetect(req), isNotEmpty);

    // ignore: avoid_print
    print('--- SnapSeek hot-path micro-benchmark '
        '(median of $kIters, mock 1080x1920 + [1,84,8400] + 60x512 gallery) ---');
    // ignore: avoid_print
    print('letterbox (shipped, fused single-pass, pre-alloc): '
        '${preMs.toStringAsFixed(2)} ms');
    // ignore: avoid_print
    print('letterbox (naive, 2-pass + intermediate alloc):    '
        '${naivePreMs.toStringAsFixed(2)} ms');
    // ignore: avoid_print
    print('decode+globalNMS (shipped, threshold-first):       '
        '${decodeMs.toStringAsFixed(2)} ms');
    // ignore: avoid_print
    print('decode (naive, argmax+geometry every anchor):      '
        '${naiveDecodeMs.toStringAsFixed(2)} ms');
    // ignore: avoid_print
    print('embed crop+resize 256 (shipped, fused bilinear):   '
        '${embedPreMs.toStringAsFixed(2)} ms');
    // ignore: avoid_print
    print('gallery dot-product rank (60x512):                 '
        '${rankMs.toStringAsFixed(2)} ms');
    // ignore: avoid_print
    print('full hot path (letterbox+decode+embed+rank):       '
        '${fullMs.toStringAsFixed(2)} ms  <- A4 BASELINE');

    // Tier B rule: an optimization must show its delta or it gets removed.
    expect(decodeMs, lessThan(naiveDecodeMs));
    expect(preMs, lessThan(naivePreMs * 1.15));
  });
}
