import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:voxscribe/services/detokenizer.dart';

/// A13 — byte-level BPE detokenization. id->token->byte decode, `Ġ`->space;
/// specials (EOT 50257, SOT 50258, pad 50256) skipped and never emitted.
void main() {
  late Detokenizer detok;

  setUpAll(() {
    detok = Detokenizer.fromVocabJson(
        File('assets/vocab.json').readAsStringSync());
  });

  test('Ġ-prefixed tokens decode with leading spaces', () {
    // Ġthe=264, Ġworld=1002 -> " the world".
    expect(detok.decode(<int>[264, 1002]), ' the world');
    // Ġhello=7751, Ġroute=7955 -> " hello route".
    expect(detok.decode(<int>[7751, 7955]), ' hello route');
  });

  test('non-prefixed token + punctuation', () {
    // Hello=15947, .=13 -> "Hello."
    expect(detok.decode(<int>[15947, 13]), 'Hello.');
  });

  test('special tokens are skipped', () {
    // SOT(50258), Ġhello(7751), EOT(50257), pad(50256) -> only " hello".
    expect(detok.decode(<int>[50258, 7751, 50257, 50256]), ' hello');
    // All-special sequence -> empty.
    expect(detok.decode(<int>[50258, 50257, 50256]), '');
  });
}
