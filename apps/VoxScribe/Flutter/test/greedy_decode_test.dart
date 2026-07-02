import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:voxscribe/services/postprocessor.dart';

/// A11 — greedy decode: idx-1 logit-row indexing, SOT seeding, EOT termination.
/// A scripted fake decoder returns fixed [maxLen, vocab] logits; we use small
/// dims to keep the buffer tiny.
void main() {
  test('reads row idx-1, collects scripted tokens, stops on EOT', () {
    const int vocab = 8, maxLen = 8, eot = 7;
    // Row r argmaxes to script[r]: [3, 5, EOT].
    final List<int> script = <int>[3, 5, eot];
    final Float32List logits = Float32List(maxLen * vocab);
    for (int r = 0; r < script.length; r++) {
      logits[r * vocab + script[r]] = 10.0;
    }
    Float32List step(Int32List ids, Int32List mask) => logits;

    final List<int> out = greedyDecode(step,
        maxLength: maxLen, vocab: vocab, sot: 6, eot: eot, pad: 0);
    // Must read row 0 then row 1 (idx-1), stop at EOT in row 2.
    expect(out, <int>[3, 5]);
  });

  test('SOT seeding: idx starts at 1 (first read is row 0); EOT-in-row-0 -> []',
      () {
    const int vocab = 8, maxLen = 8, eot = 7;
    final Float32List logits = Float32List(maxLen * vocab);
    logits[0 * vocab + eot] = 10.0; // row 0 argmax = EOT
    Float32List step(Int32List ids, Int32List mask) => logits;
    final List<int> out = greedyDecode(step,
        maxLength: maxLen, vocab: vocab, sot: 6, eot: eot, pad: 0);
    expect(out, isEmpty);
  });

  test('repetition guard caps trailing-silence token loops', () {
    const int vocab = 4, maxLen = 20, eot = 3, tok = 2;
    final Float32List logits = Float32List(maxLen * vocab);
    for (int r = 0; r < maxLen; r++) {
      logits[r * vocab + tok] = 5.0; // every row argmaxes to the same token
    }
    Float32List step(Int32List ids, Int32List mask) => logits;
    final List<int> out = greedyDecode(step,
        maxLength: maxLen,
        vocab: vocab,
        sot: 1,
        eot: eot,
        pad: 0,
        repetitionGuard: 5);
    expect(out.length, 5); // stops after 5 identical tokens
    expect(out.every((int t) => t == tok), isTrue);
  });
}
