import 'dart:typed_data';

import '../models/speaker_segment.dart';

/// Segmentation + Whisper post-processing (pure Dart, exact per SPEC).
///
/// pyannote/segmentation-3.0 powerset head: 3 local speakers, max 2
/// simultaneous => 7 classes. The model output is `[1,589,7]` log-softmax.

// Frame -> time map constants (SPEC).
const int kSegFrames = 589;
const int kSegClasses = 7;
const double kFrameScale = 270.0 / 16000.0; // 0.016875 s/frame
const double kFrameOffset = 991.0 / 16000.0 * 0.5; // 0.0309688 s
const double kOnset = 0.5;
const double kOffset = 0.5;
const double kMinDurationOn = 0.30;
const double kMinDurationOff = 0.50;

// Whisper decode constants (SPEC / shipping client ground truth).
const int kSot = 50258;
const int kEot = 50257;
const int kPad = 50256;
const int kVocab = 51865;
const int kMaxLen = 448;

/// Powerset class -> set of LOCAL speakers (0..2). 7 classes, NOT 7 speakers:
///   0:{} 1:{0} 2:{1} 3:{2} 4:{0,1} 5:{0,2} 6:{1,2}.
const List<List<int>> kPowersetTable = <List<int>>[
  <int>[], // 0 silence
  <int>[0], // 1
  <int>[1], // 2
  <int>[2], // 3
  <int>[0, 1], // 4 overlap
  <int>[0, 2], // 5 overlap
  <int>[1, 2], // 6 overlap
];

const int kNumLocalSpeakers = 3;

/// Frame index -> segment time in seconds.
double frameToTime(int frame) => frame * kFrameScale + kFrameOffset;

/// Argmax over [length] values starting at [offset].
int argmaxRange(Float32List a, int offset, int length) {
  int best = 0;
  double bestVal = a[offset];
  for (int i = 1; i < length; i++) {
    final double v = a[offset + i];
    if (v > bestVal) {
      bestVal = v;
      best = i;
    }
  }
  return best;
}

/// Per-frame argmax over 7 powerset classes -> binary activity matrix
/// labels[frame][localSpeaker]. Input is the flattened `[1,589,7]` tensor
/// (log-softmax; argmax is invariant to the monotone exp, so we argmax the raw
/// row directly — SPEC log-softmax note).
List<List<bool>> powersetDecode(
  Float32List logits, {
  int nFrames = kSegFrames,
  int nClasses = kSegClasses,
}) {
  final List<List<bool>> labels = List<List<bool>>.generate(
    nFrames,
    (_) => List<bool>.filled(kNumLocalSpeakers, false),
    growable: false,
  );
  for (int f = 0; f < nFrames; f++) {
    final int cls = argmaxRange(logits, f * nClasses, nClasses);
    for (final int spk in kPowersetTable[cls]) {
      labels[f][spk] = true;
    }
  }
  return labels;
}

/// Onset/offset state machine per local speaker over the binary activity
/// matrix, then merge runs separated by <= min_duration_off and drop runs
/// shorter than min_duration_on. Returns segments sorted by start time.
List<SpeakerSegment> onsetOffsetSegments(List<List<bool>> labels) {
  final List<SpeakerSegment> out = <SpeakerSegment>[];
  final int nFrames = labels.length;
  for (int spk = 0; spk < kNumLocalSpeakers; spk++) {
    // 1) collect raw active runs (frame ranges).
    final List<List<int>> runs = <List<int>>[]; // [startFrame, endFrameExcl]
    int? runStart;
    for (int f = 0; f < nFrames; f++) {
      final bool active = labels[f][spk];
      if (active && runStart == null) {
        runStart = f;
      } else if (!active && runStart != null) {
        runs.add(<int>[runStart, f]);
        runStart = null;
      }
    }
    if (runStart != null) runs.add(<int>[runStart, nFrames]);
    if (runs.isEmpty) continue;

    // 2) convert to time; the run covers frames [start, end-1], i.e. up to the
    //    onset time of frame `end`.
    final List<List<double>> times = <List<double>>[
      for (final List<int> r in runs) <double>[frameToTime(r[0]), frameToTime(r[1])],
    ];

    // 3) merge gaps <= min_duration_off.
    final List<List<double>> merged = <List<double>>[times.first];
    for (int i = 1; i < times.length; i++) {
      final List<double> prev = merged.last;
      final List<double> cur = times[i];
      if (cur[0] - prev[1] <= kMinDurationOff) {
        prev[1] = cur[1];
      } else {
        merged.add(cur);
      }
    }

    // 4) drop runs shorter than min_duration_on.
    for (final List<double> m in merged) {
      if (m[1] - m[0] >= kMinDurationOn) {
        out.add(SpeakerSegment(start: m[0], end: m[1], speaker: spk));
      }
    }
  }
  out.sort((SpeakerSegment a, SpeakerSegment b) => a.start.compareTo(b.start));
  return out;
}

/// One decoder step: given the current 448-long ids and attention mask, returns
/// the flattened `[1,448,51865]` logits. The real implementation wraps the
/// Melange decoder model; tests inject a scripted fake.
typedef DecoderStep = Float32List Function(Int32List ids, Int32List mask);

/// Greedy 448-step Whisper decode (SPEC steps 6-8). Seeds ids[0]=SOT, mask[0]=1,
/// idx=1; each step reads logit row `(idx-1)*vocab .. idx*vocab`, argmaxes,
/// stops on EOT. Returns the collected (non-special) token ids.
///
/// [repetitionGuard]: if the same token is emitted this many times in a row,
/// stop early. This is the worker-side mitigation for the 30 s silence-pad
/// hallucination/looping risk (Tier C): padding short spans to 30 s can make
/// Whisper loop tokens in trailing silence. EOT termination handles the normal
/// case; this guard caps pathological loops. Set <= 0 to disable.
List<int> greedyDecode(
  DecoderStep step, {
  int maxLength = kMaxLen,
  int sot = kSot,
  int eot = kEot,
  int pad = kPad,
  int vocab = kVocab,
  int repetitionGuard = 24,
}) {
  final Int32List ids = Int32List(maxLength)..fillRange(0, maxLength, pad);
  final Int32List mask = Int32List(maxLength); // zero-filled
  ids[0] = sot;
  mask[0] = 1;

  final List<int> generated = <int>[];
  int idx = 1;
  int lastTok = -1;
  int repeats = 0;
  while (idx < maxLength) {
    final Float32List logits = step(ids, mask);
    final int next = argmaxRange(logits, (idx - 1) * vocab, vocab);
    if (next == eot) break;

    if (repetitionGuard > 0) {
      if (next == lastTok) {
        repeats++;
        if (repeats >= repetitionGuard) break;
      } else {
        repeats = 0;
        lastTok = next;
      }
    }

    ids[idx] = next;
    mask[idx] = 1;
    generated.add(next);
    idx++;
  }
  return generated;
}
