import 'dart:ui';

import '../models/read_field.dart';

/// One recognized field going into PII classification.
class PiiInputField {
  const PiiInputField({
    required this.bbox,
    required this.text,
    required this.confidence,
  });

  final Rect bbox;
  final String text;
  final double confidence;
}

/// Pure-Dart PII heuristics over recognized document fields.
///
/// Text-only rules (per field): MRZ (`<<` runs / high `<` density over an
/// [A-Z0-9<] charset), dates (several common formats -> DOB class), and
/// ID-number shapes (SSN-like, passport-like, long digit runs, dense
/// alphanumeric codes).
///
/// Keyword-anchored rules (cross-field geometry): a label field such as
/// "Name", "DOB" or "ID No." marks either its own trailing value
/// ("Name: JOHN DOE") or the geometrically adjacent value field (same line to
/// the right, or directly below). Labels themselves are not redacted — only
/// values. Full reading-line grouping is intentionally out of scope for v1
/// (GATE-2 ruling); adjacency pairing covers the label->value case.
class PiiClassifier {
  static final RegExp _mrzCharset = RegExp(r'^[A-Z0-9<\s]+$');

  static final List<RegExp> _dateRes = [
    RegExp(r'\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b'),
    RegExp(r'\b\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}\b'),
    RegExp(
        r'\b\d{1,2}\s?(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\.?,?\s?\d{2,4}\b',
        caseSensitive: false),
    RegExp(
        r'\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\.?\s\d{1,2},?\s?\d{2,4}\b',
        caseSensitive: false),
  ];

  static final List<RegExp> _idRes = [
    RegExp(r'\b\d{3}-\d{2}-\d{4}\b'), // SSN-like
    RegExp(r'\b[A-Z]{1,2}\d{6,9}\b'), // passport-like
    RegExp(r'\b\d{6,}\b'), // long digit run
    RegExp(r'\b(?=[A-Z0-9-]*\d{4,})[A-Z0-9][A-Z0-9-]{7,}\b'), // dense code
  ];

  static final RegExp _nameAnchor = RegExp(
      r'^(sur\s?name|given\s?names?|first\s?name|last\s?name|full\s?name|name|holder|patient)\b\s*[:.]?\s*(.*)$',
      caseSensitive: false);
  static final RegExp _dobAnchor = RegExp(
      r'^(date\s?of\s?birth|birth\s?date|dob|born|birth)\b\s*[:.]?\s*(.*)$',
      caseSensitive: false);
  static final RegExp _idAnchor = RegExp(
      r'^(id\s?(no|number)?|passport\s?(no|number)?|document\s?(no|number)?|licen[cs]e\s?(no|number)?|mrn|ssn|personal\s?(no|number))\b\s*[:.#]?\s*(.*)$',
      caseSensitive: false);

  /// Classifies each field's text in isolation (no geometry).
  PiiClass classifyText(String text) {
    final t = text.trim();
    if (t.isEmpty) return PiiClass.other;

    // MRZ: machine-readable-zone lines are uppercase [A-Z0-9<] with filler
    // '<' runs. Strongest, most distinctive signal — check first.
    final ltCount = '<'.allMatches(t).length;
    if ((t.contains('<<') || ltCount >= 4) &&
        t.length >= 8 &&
        _mrzCharset.hasMatch(t)) {
      return PiiClass.mrz;
    }

    for (final re in _dateRes) {
      if (re.hasMatch(t)) return PiiClass.dob;
    }
    for (final re in _idRes) {
      if (re.hasMatch(t)) return PiiClass.idNumber;
    }
    return PiiClass.other;
  }

  /// Classifies a set of recognized fields, using text rules plus
  /// keyword-anchor adjacency. Returns one [ReadField] per input, in order.
  List<ReadField> classify(List<PiiInputField> fields) {
    final classes = List<PiiClass>.generate(
        fields.length, (i) => classifyText(fields[i].text));

    for (var i = 0; i < fields.length; i++) {
      final text = fields[i].text.trim();
      if (text.isEmpty) continue;

      final (anchorClass, inlineValue) = _matchAnchor(text);
      if (anchorClass == null) continue;

      if (inlineValue != null && inlineValue.trim().length >= 2) {
        // "Name: JOHN DOE" — the field carries its own value.
        classes[i] = anchorClass;
        continue;
      }

      // Label-only field: mark the geometrically adjacent value field.
      final vi = _findValueField(fields, i);
      if (vi != null && classes[vi] == PiiClass.other) {
        classes[vi] = anchorClass;
      }
    }

    return List<ReadField>.generate(
      fields.length,
      (i) => ReadField(
        bbox: fields[i].bbox,
        text: fields[i].text,
        confidence: fields[i].confidence,
        piiClass: classes[i],
      ),
    );
  }

  /// Returns (class, inline value after the label) when [text] starts with a
  /// PII label keyword, else (null, null).
  (PiiClass?, String?) _matchAnchor(String text) {
    final dob = _dobAnchor.firstMatch(text);
    if (dob != null) return (PiiClass.dob, dob.group(dob.groupCount));
    final id = _idAnchor.firstMatch(text);
    if (id != null) return (PiiClass.idNumber, id.group(id.groupCount));
    final name = _nameAnchor.firstMatch(text);
    if (name != null) return (PiiClass.name, name.group(name.groupCount));
    return (null, null);
  }

  /// Finds the value field for a label-only anchor: same line to the right
  /// (vertical overlap, nearest gap), else directly below (horizontal
  /// overlap, nearest gap).
  int? _findValueField(List<PiiInputField> fields, int anchorIdx) {
    final a = fields[anchorIdx].bbox;

    int? best;
    var bestGap = double.infinity;
    for (var j = 0; j < fields.length; j++) {
      if (j == anchorIdx || fields[j].text.trim().isEmpty) continue;
      final b = fields[j].bbox;
      final overlapV = _overlap(a.top, a.bottom, b.top, b.bottom);
      final minH = a.height < b.height ? a.height : b.height;
      if (overlapV >= 0.5 * minH && b.left >= a.left) {
        final gap = b.left - a.right;
        if (gap > -0.5 * a.width && gap < 3 * a.height && gap < bestGap) {
          bestGap = gap;
          best = j;
        }
      }
    }
    if (best != null) return best;

    bestGap = double.infinity;
    for (var j = 0; j < fields.length; j++) {
      if (j == anchorIdx || fields[j].text.trim().isEmpty) continue;
      final b = fields[j].bbox;
      final overlapH = _overlap(a.left, a.right, b.left, b.right);
      final minW = a.width < b.width ? a.width : b.width;
      if (overlapH >= 0.3 * minW && b.top >= a.bottom - 0.2 * a.height) {
        final gap = b.top - a.bottom;
        if (gap < 1.5 * a.height && gap < bestGap) {
          bestGap = gap;
          best = j;
        }
      }
    }
    return best;
  }

  double _overlap(double a1, double a2, double b1, double b2) {
    final lo = a1 > b1 ? a1 : b1;
    final hi = a2 < b2 ? a2 : b2;
    return hi - lo;
  }
}
