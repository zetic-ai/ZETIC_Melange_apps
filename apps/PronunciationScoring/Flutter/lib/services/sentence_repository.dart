import 'dart:convert';

import 'package:flutter/services.dart' show rootBundle;

import '../models/sentence.dart';

/// Loads the bundled practice sentences from assets/sentences.json.
class SentenceRepository {
  const SentenceRepository();

  static const String _asset = 'assets/sentences.json';

  Future<List<PracticeSentence>> load() async {
    final raw = await rootBundle.loadString(_asset);
    return parse(raw);
  }

  /// Parse the asset JSON string (exposed for tests).
  static List<PracticeSentence> parse(String raw) {
    final json = jsonDecode(raw) as Map<String, dynamic>;
    final list = json['sentences'] as List;
    return list
        .map((e) => PracticeSentence.fromJson(e as Map<String, dynamic>))
        .toList();
  }
}
