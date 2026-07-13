import 'dart:typed_data';

/// Recognizer output is [1, 40, 438]: 40 CTC time-steps x 438 classes.
const int kCtcSteps = 40;
const int kCtcClasses = 438;

/// CTC blank is class 0; space is the LAST class (437).
const int kCtcBlankIndex = 0;

/// One decoded recognizer result.
class RecognizedText {
  const RecognizedText({required this.text, required this.confidence});

  final String text;

  /// Mean of the per-step max probabilities over the steps that emitted a
  /// character (raw values — the head is already Softmax'd in-graph, so no
  /// extra activation is applied). 0 when no character was emitted.
  final double confidence;
}

/// Greedy CTC decoder for the PP-OCRv5 English recognizer.
///
/// The label list is EXACTLY `[blank] + en_dict.txt (436 chars, classes
/// 1..436) + ' ' (class 437)` — SPEC-binding. Decoding is per-step argmax over
/// the LAST axis (438 classes), then collapse consecutive duplicate classes,
/// THEN drop blanks (so a genuine double letter survives when separated by a
/// blank).
class CtcDecoder {
  CtcDecoder(List<String> dictChars)
      : labels = List.unmodifiable(['', ...dictChars, ' ']) {
    if (labels.length != kCtcClasses) {
      throw ArgumentError(
          'CTC label list must have $kCtcClasses entries '
          '([blank] + dict + space), got ${labels.length} '
          '(dict had ${dictChars.length}, expected ${kCtcClasses - 2})');
    }
  }

  /// Builds the decoder from the raw contents of en_dict.txt (one character
  /// per line; line i is class i+1).
  factory CtcDecoder.fromDictString(String raw) {
    final lines = raw.split('\n');
    // Drop a single trailing empty line from a terminating newline; real dict
    // lines are never empty (each holds exactly one character).
    if (lines.isNotEmpty && lines.last.isEmpty) lines.removeLast();
    final chars = lines
        .map((l) => l.endsWith('\r') ? l.substring(0, l.length - 1) : l)
        .toList(growable: false);
    return CtcDecoder(chars);
  }

  /// labels[0] is the blank (empty string); labels[437] is ' '.
  final List<String> labels;

  /// Decodes a flattened [1, 40, 438] probability tensor.
  RecognizedText decode(Float32List probs) {
    assert(probs.length == kCtcSteps * kCtcClasses);

    final sb = StringBuffer();
    var confSum = 0.0;
    var emitted = 0;
    var prevClass = -1;

    for (var t = 0; t < kCtcSteps; t++) {
      final int base = t * kCtcClasses;
      // Argmax over the LAST axis (classes) for this step.
      var best = 0;
      var bestP = probs[base];
      for (var c = 1; c < kCtcClasses; c++) {
        final double p = probs[base + c];
        if (p > bestP) {
          bestP = p;
          best = c;
        }
      }

      // Collapse consecutive duplicates first...
      if (best == prevClass) {
        continue;
      }
      prevClass = best;

      // ...then drop blanks.
      if (best == kCtcBlankIndex) continue;

      sb.write(labels[best]);
      confSum += bestP;
      emitted++;
    }

    return RecognizedText(
      text: sb.toString(),
      confidence: emitted > 0 ? confSum / emitted : 0.0,
    );
  }
}
