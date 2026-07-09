import 'dart:math' as math;
import 'dart:typed_data';

/// Exact input contract for citrinet256_phoneme.onnx: 81760 mono samples at
/// 16 kHz (5.11 s), float32 in [-1, 1], raw waveform (the mel frontend is baked
/// into the ONNX — no normalization here).
const int kTargetSamples = 81760;
const int kTargetRate = 16000;

/// The only two mic stream rates we accept.
const int kNativeRate = 16000;
const int kDecimateRate = 48000; // integer /3 down to 16 kHz.

/// RMS of the noise used to pad an OS-truncated capture. NEVER pad with zeros:
/// runs of digital zeros hit the graph's per-utterance normalization and
/// measurably wreck accuracy (PER 0.29 -> 0.58). ~1e-3 RMS room-tone proxy.
const double kPadRms = 1e-3;

/// Longest run of exact zeros the preprocessor is allowed to emit. Used only as
/// a defensive contract check (the noise-pad path never produces zeros).
const int kMaxZeroRun = 160;

/// How the reported mic rate was handled — surfaced on the HUD.
enum RateMode { native16k, decimate48k }

/// Thrown when the mic delivers a rate we refuse to silently resample.
class SampleRateException implements Exception {
  const SampleRateException(this.reportedRate);
  final int reportedRate;
  @override
  String toString() =>
      'Unsupported mic sample rate $reportedRate Hz '
      '(accept only $kNativeRate or $kDecimateRate)';
}

/// Resolve a reported mic rate to a handling mode, or refuse. Never resamples a
/// rate other than the two explicitly supported ones.
RateMode resolveRate(int reportedRate) {
  switch (reportedRate) {
    case kNativeRate:
      return RateMode.native16k;
    case kDecimateRate:
      return RateMode.decimate48k;
    default:
      throw SampleRateException(reportedRate);
  }
}

/// Human-readable HUD note for a resolved rate mode.
String rateModeLabel(RateMode mode) {
  switch (mode) {
    case RateMode.native16k:
      return '16k native';
    case RateMode.decimate48k:
      return '48k→16k decimation active';
  }
}

/// Little-endian PCM16 bytes -> float32 samples in [-1, 1] via /32768.0.
/// Edge values map to +32767/32768 and -32768/32768 = -1.0 exactly.
Float32List pcm16ToFloat32(Uint8List bytes) {
  // A trailing odd byte (partial frame) is ignored.
  final n = bytes.length ~/ 2;
  final view = ByteData.view(bytes.buffer, bytes.offsetInBytes, n * 2);
  final out = Float32List(n);
  for (var i = 0; i < n; i++) {
    out[i] = view.getInt16(i * 2, Endian.little) / 32768.0;
  }
  return out;
}

// 31-tap Hamming-windowed-sinc low-pass, cutoff 7.5 kHz at 48 kHz (safely below
// the 8 kHz post-decimation Nyquist). Computed once, reused per capture.
final Float32List _decimationTaps = _buildLowPass(31, 7500.0 / 48000.0);

Float32List _buildLowPass(int numTaps, double fc) {
  final taps = Float32List(numTaps);
  final m = (numTaps - 1) / 2.0;
  var sum = 0.0;
  for (var i = 0; i < numTaps; i++) {
    final x = i - m;
    final sinc = x == 0 ? 2 * fc : math.sin(2 * math.pi * fc * x) / (math.pi * x);
    final hamming = 0.54 - 0.46 * math.cos(2 * math.pi * i / (numTaps - 1));
    taps[i] = sinc * hamming;
    sum += taps[i];
  }
  for (var i = 0; i < numTaps; i++) {
    taps[i] /= sum; // unity DC gain
  }
  return taps;
}

/// Decimate a 48 kHz signal to 16 kHz: anti-alias low-pass THEN take every 3rd
/// sample. This is a proper decimation — NOT naive interpolation/subsampling,
/// which would alias every formant above 8 kHz back into the band.
Float32List decimate48to16(Float32List input) {
  final taps = _decimationTaps;
  final m = (taps.length - 1) ~/ 2;
  final outLen = input.length ~/ 3;
  final out = Float32List(outLen);
  for (var o = 0; o < outLen; o++) {
    final center = o * 3;
    var acc = 0.0;
    for (var k = 0; k < taps.length; k++) {
      final idx = center + k - m;
      if (idx >= 0 && idx < input.length) {
        acc += taps[k] * input[idx];
      }
    }
    out[o] = acc;
  }
  return out;
}

/// Fit [samples16k] to exactly [kTargetSamples]: truncate if long, or copy and
/// pad the tail with ~[kPadRms] RMS noise if short. NEVER zero-pads.
Float32List fitToWindow(Float32List samples16k, {math.Random? rng}) {
  final out = Float32List(kTargetSamples);
  final n = math.min(samples16k.length, kTargetSamples);
  for (var i = 0; i < n; i++) {
    out[i] = samples16k[i];
  }
  if (n < kTargetSamples) {
    final r = rng ?? math.Random(7);
    // Uniform[-a, a] has RMS a/sqrt(3); solve for target RMS.
    final a = kPadRms * math.sqrt(3);
    for (var i = n; i < kTargetSamples; i++) {
      out[i] = (r.nextDouble() * 2 - 1) * a;
    }
  }
  return out;
}

/// Full pipeline: raw PCM16 capture bytes at [reportedRate] -> the exact
/// float32[81760] model input. Refuses unsupported rates (never silent-resamples).
Float32List buildModelInput(Uint8List pcmBytes, int reportedRate,
    {math.Random? rng}) {
  final mode = resolveRate(reportedRate);
  var samples = pcm16ToFloat32(pcmBytes);
  if (mode == RateMode.decimate48k) {
    samples = decimate48to16(samples);
  }
  return fitToWindow(samples, rng: rng);
}

/// Length of the longest run of EXACT zeros in [x] (defensive contract check).
int longestZeroRun(Float32List x) {
  var best = 0;
  var run = 0;
  for (var i = 0; i < x.length; i++) {
    if (x[i] == 0.0) {
      run++;
      if (run > best) best = run;
    } else {
      run = 0;
    }
  }
  return best;
}
