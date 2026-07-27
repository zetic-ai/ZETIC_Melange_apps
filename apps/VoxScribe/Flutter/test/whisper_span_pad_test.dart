import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:voxscribe/services/log_mel.dart';
import 'package:voxscribe/services/preprocessor.dart';

/// A10 — 30 s span framing. Every span is zero-padded/truncated to 480000
/// samples (30 s) before log-mel, yielding exactly 3000 frames.
void main() {
  test('2 s span zero-pads to 480000 and yields 3000 mel frames', () {
    final Float32List span = Float32List(32000)..fillRange(0, 32000, 0.4);
    final Float32List padded = padSpanTo480000(span);
    expect(padded.length, kWhisperSpanSamples); // 480000
    expect(padded[0], closeTo(0.4, 1e-6));
    expect(padded[31999], closeTo(0.4, 1e-6));
    expect(padded[32000], 0.0); // pad region
    expect(LogMel.frameCountFor(padded.length), 3000);
  });

  test('long span truncates to 480000', () {
    final Float32List span = Float32List(600000)..fillRange(0, 600000, 0.2);
    final Float32List fitted = padSpanTo480000(span);
    expect(fitted.length, kWhisperSpanSamples);
    expect(fitted[479999], closeTo(0.2, 1e-6));
  });

  test('whisperSpan slices then fits to 480000', () {
    final Float32List mono = Float32List(160000);
    for (int i = 0; i < mono.length; i++) {
      mono[i] = 1.0;
    }
    // Slice [16000, 48000) = 2 s of ones.
    final Float32List span = whisperSpan(mono, 16000, 48000);
    expect(span.length, kWhisperSpanSamples);
    expect(span[0], closeTo(1.0, 1e-6));
    expect(span[31999], closeTo(1.0, 1e-6));
    expect(span[32000], 0.0);
  });
}
