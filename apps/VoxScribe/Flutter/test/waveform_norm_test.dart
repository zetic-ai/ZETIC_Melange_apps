import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:voxscribe/services/preprocessor.dart';

/// A3 — waveform normalization trap. int16 -> /32768 and NOTHING else; float
/// PCM passes through with no double-normalization.
void main() {
  test('int16 extremes divide by 32768 into [-1,1]', () {
    final Int16List pcm = Int16List.fromList(<int>[32767, 0, -32768]);
    final Float32List f = int16ToFloat(pcm);
    expect(f[0], closeTo(32767 / 32768.0, 1e-9)); // ~+0.99997
    expect(f[1], closeTo(0.0, 1e-12));
    expect(f[2], closeTo(-1.0, 1e-12)); // -32768/32768 = -1.0
    for (final double v in f) {
      expect(v, inInclusiveRange(-1.0, 1.0));
    }
  });

  test('decodeWav of a float32 WAV passes samples through unchanged', () {
    final Float32List src =
        Float32List.fromList(<double>[0.25, -0.5, 0.0, 0.75]);
    final Uint8List wav = _floatWav(src, 16000, 1);
    final DecodedAudio a = decodeWav(wav);
    expect(a.sampleRate, 16000);
    expect(a.channels, 1);
    for (int i = 0; i < src.length; i++) {
      expect(a.samples[i], closeTo(src[i], 1e-6)); // no extra scaling
    }
  });
}

/// Builds a minimal IEEE-float32 (fmt 3) mono WAV for the decode test.
Uint8List _floatWav(Float32List samples, int rate, int channels) {
  final int dataLen = samples.length * 4;
  final ByteData bd = ByteData(44 + dataLen);
  void tag(int o, String s) {
    for (int i = 0; i < 4; i++) {
      bd.setUint8(o + i, s.codeUnitAt(i));
    }
  }

  tag(0, 'RIFF');
  bd.setUint32(4, 36 + dataLen, Endian.little);
  tag(8, 'WAVE');
  tag(12, 'fmt ');
  bd.setUint32(16, 16, Endian.little);
  bd.setUint16(20, 3, Endian.little); // IEEE float
  bd.setUint16(22, channels, Endian.little);
  bd.setUint32(24, rate, Endian.little);
  bd.setUint32(28, rate * channels * 4, Endian.little);
  bd.setUint16(32, channels * 4, Endian.little);
  bd.setUint16(34, 32, Endian.little);
  tag(36, 'data');
  bd.setUint32(40, dataLen, Endian.little);
  for (int i = 0; i < samples.length; i++) {
    bd.setFloat32(44 + i * 4, samples[i], Endian.little);
  }
  return bd.buffer.asUint8List();
}
