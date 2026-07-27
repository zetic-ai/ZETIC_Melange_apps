import 'dart:convert';

/// Pure-Dart GPT-2 byte-level BPE detokenizer for Whisper token ids.
///
/// The native SDK `WhisperWrapper.decodeToken` is not exposed to Flutter
/// (GATE-2 decision 3), so we port the byte-level decode:
///   id -> token string (via bundled `vocab.json`, token->id)
///        -> map each char back to a byte (reverse of GPT-2 `bytes_to_unicode`)
///        -> UTF-8 decode the assembled byte stream.
///
/// Special / non-vocab ids are skipped: any id >= 50257 (EOT 50257, pad 50256
/// is < 50257 so guarded explicitly, SOT 50258, and all timestamp/special ids
/// >= 50258 which have no `vocab.json` entry). With SOT-only seeding these
/// should not appear in the collected ids, but we guard regardless.
class Detokenizer {
  Detokenizer._(this._idToToken, this._byteDecoder);

  /// id -> token unicode string (null where there is no vocab entry).
  final List<String?> _idToToken;

  /// reverse of bytes_to_unicode: unicode code unit -> raw byte (0..255).
  final Map<int, int> _byteDecoder;

  static const int eot = 50257;
  static const int pad = 50256;
  static const int sot = 50258;

  /// Builds a detokenizer from the raw `vocab.json` contents (token -> id).
  factory Detokenizer.fromVocabJson(String vocabJson) {
    final Map<String, dynamic> map =
        json.decode(vocabJson) as Map<String, dynamic>;
    int maxId = 0;
    map.forEach((_, dynamic v) {
      final int id = v as int;
      if (id > maxId) maxId = id;
    });
    final List<String?> idToToken = List<String?>.filled(maxId + 1, null);
    map.forEach((String token, dynamic v) {
      idToToken[v as int] = token;
    });
    return Detokenizer._(idToToken, _buildByteDecoder());
  }

  /// Decodes a sequence of token ids into text. Special ids are skipped.
  String decode(List<int> ids) {
    final List<int> bytes = <int>[];
    for (final int id in ids) {
      if (id >= eot || id == pad || id == sot) continue; // specials
      if (id < 0 || id >= _idToToken.length) continue;
      final String? token = _idToToken[id];
      if (token == null) continue;
      for (final int cu in token.runes) {
        final int? b = _byteDecoder[cu];
        if (b != null) bytes.add(b);
      }
    }
    if (bytes.isEmpty) return '';
    return utf8.decode(bytes, allowMalformed: true);
  }

  /// GPT-2 bytes_to_unicode, reversed (unicode char code -> byte). Mirrors
  /// the reference `bytes_to_unicode()` used by tiktoken/transformers.
  static Map<int, int> _buildByteDecoder() {
    final List<int> bs = <int>[];
    for (int i = '!'.codeUnitAt(0); i <= '~'.codeUnitAt(0); i++) {
      bs.add(i);
    }
    for (int i = 0xA1; i <= 0xAC; i++) {
      bs.add(i);
    }
    for (int i = 0xAE; i <= 0xFF; i++) {
      bs.add(i);
    }
    final List<int> cs = List<int>.from(bs);
    int n = 0;
    for (int b = 0; b < 256; b++) {
      if (!bs.contains(b)) {
        bs.add(b);
        cs.add(256 + n);
        n++;
      }
    }
    final Map<int, int> decoder = <int, int>{};
    for (int i = 0; i < bs.length; i++) {
      decoder[cs[i]] = bs[i]; // unicode code unit -> raw byte
    }
    return decoder;
  }
}
