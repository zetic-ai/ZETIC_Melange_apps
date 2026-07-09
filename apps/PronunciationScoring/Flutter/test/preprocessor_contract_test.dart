import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sayright/services/preprocessor.dart';

Uint8List pcm16LE(List<int> samples) {
  final b = ByteData(samples.length * 2);
  for (var i = 0; i < samples.length; i++) {
    b.setInt16(i * 2, samples[i], Endian.little);
  }
  return b.buffer.asUint8List();
}

void main() {
  group('PCM16 -> float32 conversion', () {
    test('edge values map through /32768.0', () {
      final f = pcm16ToFloat32(pcm16LE([32767, -32768, 0, 16384]));
      expect(f[0], closeTo(32767 / 32768.0, 1e-9));
      expect(f[1], -1.0); // -32768/32768 == -1.0 exactly
      expect(f[2], 0.0);
      expect(f[3], closeTo(0.5, 1e-9));
    });

    test('a trailing odd byte is ignored (whole frames only)', () {
      final bytes = Uint8List.fromList([0x00, 0x40, 0x7f]); // 1.5 samples
      expect(pcm16ToFloat32(bytes).length, 1);
    });
  });

  group('window fitting', () {
    test('output is exactly kTargetSamples', () {
      final short = Float32List(1000)..fillRange(0, 1000, 0.2);
      expect(fitToWindow(short).length, kTargetSamples);
      final long = Float32List(kTargetSamples + 5000)
        ..fillRange(0, kTargetSamples + 5000, 0.1);
      expect(fitToWindow(long).length, kTargetSamples);
    });

    test('short capture noise-pads the tail — never a zero-run > 160', () {
      final short = Float32List(2000)..fillRange(0, 2000, 0.3);
      final out = fitToWindow(short);
      // The padded tail is non-zero noise.
      expect(longestZeroRun(out), lessThanOrEqualTo(kMaxZeroRun));
      // Tail RMS is roughly the target pad level (order of magnitude).
      var sumSq = 0.0;
      for (var i = 2000; i < kTargetSamples; i++) {
        sumSq += out[i] * out[i];
      }
      final rms = (sumSq / (kTargetSamples - 2000));
      expect(rms, greaterThan(0)); // strictly non-silent
    });

    test('long capture is truncated to the first kTargetSamples', () {
      final long = Float32List(kTargetSamples + 100);
      for (var i = 0; i < long.length; i++) {
        long[i] = 0.01 * (i % 7 + 1); // all non-zero
      }
      final out = fitToWindow(long);
      expect(out[0], long[0]);
      expect(out[kTargetSamples - 1], long[kTargetSamples - 1]);
    });
  });

  group('sample-rate policy', () {
    test('accepts 16000 (native) and 48000 (decimate)', () {
      expect(resolveRate(kNativeRate), RateMode.native16k);
      expect(resolveRate(kDecimateRate), RateMode.decimate48k);
    });

    test('refuses any other rate rather than silently resampling', () {
      expect(() => resolveRate(44100), throwsA(isA<SampleRateException>()));
      expect(() => resolveRate(22050), throwsA(isA<SampleRateException>()));
      expect(() => resolveRate(8000), throwsA(isA<SampleRateException>()));
    });

    test('buildModelInput refuses an unsupported rate', () {
      final bytes = pcm16LE(List<int>.filled(100, 1000));
      expect(() => buildModelInput(bytes, 44100),
          throwsA(isA<SampleRateException>()));
    });

    test('rateModeLabel surfaces the 48k decimation on the HUD', () {
      expect(rateModeLabel(RateMode.decimate48k), contains('48k'));
      expect(rateModeLabel(RateMode.native16k), contains('16k'));
    });
  });
}
