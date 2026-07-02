import 'dart:typed_data';

/// Shared audio front end (pure Dart), exact per SPEC pre-processing:
///   1. decode PCM (16-bit int or 32-bit float WAV);
///   2. if stereo, downmix to mono = channel 0 (GATE-2 decision 7);
///   3. if rate != 16000, linear-resample to 16000 (N input sec -> N output sec);
///   4. ensure float32 in [-1,1] (int16 -> /32768; float passes through).
/// Then framing helpers: 10 s segmentation window (160000 samples) and the
/// per-span 30 s pad/truncate (480000 samples) for Whisper.
class DecodedAudio {
  const DecodedAudio(this.samples, this.sampleRate, this.channels);

  /// Interleaved float PCM in [-1,1].
  final Float32List samples;
  final int sampleRate;
  final int channels;
}

const int kTargetSampleRate = 16000;
const int kSegmentationSamples = 160000; // 10 s @ 16 kHz
const int kWhisperSpanSamples = 480000; // 30 s @ 16 kHz

/// Minimal RIFF/WAVE PCM decoder. Supports PCM int16 (fmt 1) and IEEE float32
/// (fmt 3), mono or interleaved multi-channel. Returns interleaved float [-1,1].
DecodedAudio decodeWav(Uint8List bytes) {
  final ByteData bd = ByteData.sublistView(bytes);
  if (bytes.length < 12 ||
      _tag(bytes, 0) != 'RIFF' ||
      _tag(bytes, 8) != 'WAVE') {
    throw const FormatException('Not a RIFF/WAVE file');
  }
  int offset = 12;
  int audioFormat = 1, channels = 1, sampleRate = kTargetSampleRate, bits = 16;
  int dataStart = -1, dataLen = 0;
  while (offset + 8 <= bytes.length) {
    final String id = _tag(bytes, offset);
    final int size = bd.getUint32(offset + 4, Endian.little);
    final int body = offset + 8;
    if (id == 'fmt ') {
      audioFormat = bd.getUint16(body, Endian.little);
      channels = bd.getUint16(body + 2, Endian.little);
      sampleRate = bd.getUint32(body + 4, Endian.little);
      bits = bd.getUint16(body + 14, Endian.little);
    } else if (id == 'data') {
      dataStart = body;
      dataLen = size;
    }
    offset = body + size + (size.isOdd ? 1 : 0); // chunks are word-aligned
  }
  if (dataStart < 0) throw const FormatException('No data chunk');
  if (dataStart + dataLen > bytes.length) {
    dataLen = bytes.length - dataStart; // tolerate a short/again-padded tail
  }

  final Float32List out;
  if (audioFormat == 3 && bits == 32) {
    final int n = dataLen ~/ 4;
    out = Float32List(n);
    for (int i = 0; i < n; i++) {
      out[i] = bd.getFloat32(dataStart + i * 4, Endian.little);
    }
  } else if (audioFormat == 1 && bits == 16) {
    final int n = dataLen ~/ 2;
    out = Float32List(n);
    for (int i = 0; i < n; i++) {
      out[i] = bd.getInt16(dataStart + i * 2, Endian.little) / 32768.0;
    }
  } else {
    throw FormatException('Unsupported WAV: fmt=$audioFormat bits=$bits');
  }
  return DecodedAudio(out, sampleRate, channels);
}

/// Downmix interleaved [L0,R0,L1,R1,...] to mono by taking channel 0 (SPEC:
/// reference uses ch0). For mono input this is a no-op copy.
Float32List downmixCh0(Float32List interleaved, int channels) {
  if (channels <= 1) return interleaved;
  final int frames = interleaved.length ~/ channels;
  final Float32List mono = Float32List(frames);
  for (int i = 0; i < frames; i++) {
    mono[i] = interleaved[i * channels];
  }
  return mono;
}

/// Convert int16 PCM samples to float [-1,1] by dividing by 32768. Provided as
/// a standalone for the normalization trap test; [decodeWav] already does this.
Float32List int16ToFloat(Int16List pcm) {
  final Float32List out = Float32List(pcm.length);
  for (int i = 0; i < pcm.length; i++) {
    out[i] = pcm[i] / 32768.0;
  }
  return out;
}

/// Linear-interpolation resampler. N seconds in -> N seconds out:
/// outLength = round(inLength * outRate / inRate). Endpoints are preserved.
/// A no-op (copy) when rates already match.
Float32List resampleLinear(Float32List input, int inRate, int outRate) {
  if (inRate == outRate || input.isEmpty) {
    return Float32List.fromList(input);
  }
  final int outLen = (input.length * outRate / inRate).round();
  if (outLen <= 1) {
    return Float32List.fromList(input.isEmpty ? input : <double>[input.first]);
  }
  final Float32List out = Float32List(outLen);
  final double step = (input.length - 1) / (outLen - 1);
  for (int i = 0; i < outLen; i++) {
    final double pos = i * step;
    final int i0 = pos.floor();
    final int i1 = i0 + 1 < input.length ? i0 + 1 : i0;
    final double frac = pos - i0;
    out[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  out[0] = input.first;
  out[outLen - 1] = input.last; // exact endpoints
  return out;
}

/// Full shared front end: bytes -> mono, 16 kHz, float [-1,1].
Float32List preprocessToMono16k(Uint8List wavBytes) {
  final DecodedAudio a = decodeWav(wavBytes);
  final Float32List mono = downmixCh0(a.samples, a.channels);
  return resampleLinear(mono, a.sampleRate, kTargetSampleRate);
}

/// Take/zero-pad to exactly one 10 s segmentation window (160000 samples).
Float32List segmentationWindow(Float32List mono16k) =>
    _fitTo(mono16k, kSegmentationSamples);

/// Slice [startSample, endSample) of [mono16k] then zero-pad/truncate to the
/// fixed 30 s Whisper span (480000 samples).
Float32List whisperSpan(Float32List mono16k, int startSample, int endSample) {
  final int s = startSample.clamp(0, mono16k.length);
  final int e = endSample.clamp(s, mono16k.length);
  final Float32List span = Float32List(kWhisperSpanSamples);
  final int n = (e - s).clamp(0, kWhisperSpanSamples);
  for (int i = 0; i < n; i++) {
    span[i] = mono16k[s + i];
  }
  return span;
}

/// Zero-pad or truncate an arbitrary span to the fixed 30 s length.
Float32List padSpanTo480000(Float32List span) =>
    _fitTo(span, kWhisperSpanSamples);

Float32List _fitTo(Float32List src, int n) {
  final Float32List out = Float32List(n); // zero-filled
  final int copy = src.length < n ? src.length : n;
  for (int i = 0; i < copy; i++) {
    out[i] = src[i];
  }
  return out;
}

String _tag(Uint8List b, int o) =>
    String.fromCharCodes(<int>[b[o], b[o + 1], b[o + 2], b[o + 3]]);
