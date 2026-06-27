import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:voxscribe/services/log_mel.dart';

/// A9 — Whisper log-mel exactness. Output is [1,80,3000]-shaped (3000 frames for
/// a 30 s span); the clamp/scale formula `(max(log, max-8)+4)/4` must match a
/// reference vector. Here we validate the Dart STFT + mel matmul + clamp/scale
/// against a golden computed by `tool/gen_logmel_golden.py` (pure-Python
/// torch.stft-equivalent), using the SAME bundled filterbank asset.
void main() {
  late LogMel logMel;

  setUpAll(() {
    final Uint8List bytes =
        File('assets/mel_filters_80.bin').readAsBytesSync();
    final ByteData bd = ByteData.sublistView(bytes);
    final Float32List filters = Float32List(bytes.length ~/ 4);
    for (int i = 0; i < filters.length; i++) {
      filters[i] = bd.getFloat32(i * 4, Endian.little);
    }
    logMel = LogMel(filters);
  });

  test('frame count matches torch.stft(center) then drop-last', () {
    // A 30 s span (480000 samples) -> exactly 3000 frames.
    expect(LogMel.frameCountFor(480000), 3000);
    // The 1 s test input -> 100 frames.
    expect(LogMel.frameCountFor(16000), 100);
  });

  test('440 Hz sine matches golden log-mel reference vector', () {
    const int n = 16000; // 1 s @ 16 kHz
    final Float32List audio = Float32List(n);
    for (int i = 0; i < n; i++) {
      audio[i] = 0.5 * math.sin(2 * math.pi * 440.0 * i / 16000.0);
    }
    final LogMelResult r = logMel.compute(audio);
    expect(r.frames, 100);
    expect(r.data.length, 80 * 100);

    // Golden points (mel, frame) -> value, from gen_logmel_golden.py.
    const List<List<double>> golden = <List<double>>[
      <double>[0, 0, 0.983279],
      <double>[0, 50, -0.561796],
      <double>[1, 0, 0.986621],
      <double>[10, 10, 1.348738],
      <double>[20, 30, -0.561796],
      <double>[40, 50, -0.561796],
      <double>[79, 99, -0.561796],
      <double>[5, 99, 0.512655],
    ];
    for (final List<double> g in golden) {
      final int m = g[0].toInt();
      final int t = g[1].toInt();
      final double got = r.data[m * r.frames + t];
      expect(got, closeTo(g[2], 5e-3),
          reason: 'mel=$m frame=$t expected ${g[2]} got $got');
    }
  });
}
