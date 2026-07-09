import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sayright/services/preprocessor.dart';

Float32List sine(double freq, int rate, int n) {
  final x = Float32List(n);
  for (var i = 0; i < n; i++) {
    x[i] = math.sin(2 * math.pi * freq * i / rate);
  }
  return x;
}

/// Frequency from positive/negative zero-crossings (accurate for a clean sine).
double estimateFreq(Float32List x, int rate) {
  var crossings = 0;
  for (var i = 1; i < x.length; i++) {
    if ((x[i - 1] <= 0) != (x[i] <= 0)) crossings++;
  }
  return crossings / 2.0 / (x.length / rate);
}

double rms(Float32List x) {
  var s = 0.0;
  for (final v in x) {
    s += v * v;
  }
  return math.sqrt(s / x.length);
}

void main() {
  group('48 kHz -> 16 kHz decimation', () {
    test('a 400 Hz sine keeps its frequency after decimation', () {
      final at48k = sine(400, 48000, 48000); // 1 s
      final at16k = decimate48to16(at48k);
      expect(at16k.length, 16000);
      expect(estimateFreq(at16k, 16000), closeTo(400.0, 5.0));
    });

    test('output is exactly kTargetSamples after windowing', () {
      // A full 5.11 s window captured at 48 kHz.
      final at48k = sine(400, 48000, kTargetSamples * 3);
      final at16k = decimate48to16(at48k);
      expect(at16k.length, kTargetSamples);
      expect(fitToWindow(at16k).length, kTargetSamples);
    });

    test('passband tone survives, out-of-band tone is anti-alias attenuated',
        () {
      // 1 kHz is well inside the 8 kHz post-decimation band.
      final pass = decimate48to16(sine(1000, 48000, 48000));
      // 20 kHz would alias to 4 kHz if not low-passed first.
      final alias = decimate48to16(sine(20000, 48000, 48000));
      expect(rms(alias), lessThan(rms(pass) * 0.2),
          reason: 'anti-alias low-pass must suppress the 20 kHz tone');
    });

    test('decimation is not naive subsampling (which would alias 20 kHz)', () {
      // Naive every-3rd-sample of 20 kHz @48k aliases to a strong 4 kHz tone.
      final at48k = sine(20000, 48000, 48000);
      final naive = Float32List(at48k.length ~/ 3);
      for (var i = 0; i < naive.length; i++) {
        naive[i] = at48k[i * 3];
      }
      final proper = decimate48to16(at48k);
      expect(rms(proper), lessThan(rms(naive)),
          reason: 'proper decimation attenuates the alias naive keeps');
    });
  });
}
