import 'dart:typed_data';

import '../models/phonemes.dart';

/// Frame-major view over the model's flat `logprobs` output.
///
/// The model emits float32[1, 64, 45] as a flat [64*45] buffer in ROW-MAJOR
/// order: element index = frame * 45 + class. Values are log-softmax
/// (log-probabilities). Reading it class-major would silently transpose the
/// tensor (tensor_layout_test guards this).
class LogProbView {
  LogProbView(this.data, {this.frames = Phonemes.frameCount, this.classes = Phonemes.classCount})
      : assert(data.length == frames * classes,
            'expected ${frames * classes} logprobs, got ${data.length}');

  final Float32List data;
  final int frames;
  final int classes;

  /// log P(class | frame).
  double at(int frame, int cls) => data[frame * classes + cls];

  /// argmax class over one frame.
  int argmaxAt(int frame) {
    final base = frame * classes;
    var best = base;
    for (var c = base + 1; c < base + classes; c++) {
      if (data[c] > data[best]) best = c;
    }
    return best - base;
  }
}

/// Result of the frame-level decode: greedy string + blank-fraction proxy.
class GreedyDecode {
  const GreedyDecode({required this.phonemes, required this.blankFraction});

  /// Collapsed ARPABET labels ("what we heard").
  final List<String> phonemes;

  /// Fraction of frames whose argmax is the CTC blank (window-fill proxy).
  final double blankFraction;
}

/// Greedy CTC decode: per-frame argmax, collapse consecutive repeats, drop
/// blank and specials, map ids -> ARPABET.
///
/// Collapse semantics (exactly matches validate_onnx.py): a phoneme is emitted
/// only when it differs from the PREVIOUS frame's argmax id and is a real
/// phoneme (id < 39, not blank). So "L blank L" emits two L's (the blank breaks
/// the run) while "L L" emits one.
GreedyDecode greedyDecode(LogProbView lp) {
  final out = <String>[];
  var prev = -1;
  var blanks = 0;
  for (var f = 0; f < lp.frames; f++) {
    final id = lp.argmaxAt(f);
    if (id == Phonemes.blank) blanks++;
    if (id != prev && Phonemes.isPhoneme(id)) {
      out.add(Phonemes.arpabet[id]);
    }
    prev = id;
  }
  return GreedyDecode(
    phonemes: out,
    blankFraction: blanks / lp.frames,
  );
}
