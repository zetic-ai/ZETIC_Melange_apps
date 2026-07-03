import 'dart:typed_data';

import '../config.dart';

/// A decoded crop: the string plus the mean max-probability confidence.
class CtcDecodeResult {
  const CtcDecodeResult(this.text, this.confidence);

  final String text;
  final double confidence;
}

/// Greedy CTC decoder for the recognizer's [1,40,838] output.
///
/// The 838-class map is EXACTLY (SPEC-binding — NOT 438, no other app's
/// dictionary): index 0 = CTC blank, indices 1..836 = latin_charset.txt
/// lines 1..836 (order preserved), index 837 = space ' '.
class CtcDecoder {
  /// Builds the class map from the raw charset lines (836 entries). Throws
  /// [StateError] if the resulting map is not exactly [kRecNumClasses] —
  /// an off-by-one here silently shifts every decoded character.
  CtcDecoder.fromCharsetLines(List<String> charsetLines)
      : classes = ['', ...charsetLines, ' '] {
    if (classes.length != kRecNumClasses) {
      throw StateError(
        'CTC charset map must have exactly $kRecNumClasses classes '
        '(blank + 836 chars + space); got ${classes.length}. '
        'Check latin_charset.txt.',
      );
    }
  }

  /// Parses the raw charset asset text: one character per line, order
  /// preserved, trailing newline tolerated. Lines are NOT trimmed (a line
  /// could legitimately be a combining character), only the line breaks are
  /// removed.
  factory CtcDecoder.fromCharsetText(String text) {
    final lines = text.split('\n');
    while (lines.isNotEmpty && lines.last.isEmpty) {
      lines.removeLast();
    }
    return CtcDecoder.fromCharsetLines(
      [for (final l in lines) l.endsWith('\r') ? l.substring(0, l.length - 1) : l],
    );
  }

  /// classes[0] = blank (''), classes[1..836] = charset, classes[837] = ' '.
  final List<String> classes;

  /// Greedy CTC decode over a time-major [T,C] probability tensor
  /// (Softmax is BAKED into the ONNX — values are already probabilities and
  /// no activation or renormalization is applied here).
  ///
  /// Per step: argmax over the LAST axis (C=838). Then collapse consecutive
  /// duplicate indices, drop blanks (index 0), and map survivors to chars.
  /// Confidence = mean of the per-step max prob over the EMITTED steps
  /// (non-blank, deduplicated — PaddleOCR CTCLabelDecode parity).
  CtcDecodeResult decode(
    Float32List probs, {
    int timeSteps = kRecTimeSteps,
    int numClasses = kRecNumClasses,
  }) {
    assert(probs.length >= timeSteps * numClasses,
        'CTC tensor must be at least T*C');
    final buffer = StringBuffer();
    var prev = -1;
    var confSum = 0.0;
    var emitted = 0;

    for (var t = 0; t < timeSteps; t++) {
      final base = t * numClasses;
      var best = 0;
      var bestProb = probs[base];
      for (var c = 1; c < numClasses; c++) {
        final v = probs[base + c];
        if (v > bestProb) {
          bestProb = v;
          best = c;
        }
      }
      if (best != 0 && best != prev) {
        buffer.write(classes[best]);
        confSum += bestProb;
        emitted++;
      }
      prev = best;
    }

    return CtcDecodeResult(
      buffer.toString(),
      emitted > 0 ? confSum / emitted : 0.0,
    );
  }
}
