import 'dart:ui';

/// PII classes the heuristic classifier can assign to a recognized field.
enum PiiClass {
  name('NAME'),
  dob('DOB'),
  idNumber('ID'),
  mrz('MRZ'),
  other('TEXT');

  const PiiClass(this.label);

  /// Short label shown on redaction bars / HUD chips.
  final String label;

  /// Whether fields of this class get redacted in the live preview.
  bool get isPii => this != PiiClass.other;
}

/// A recognized text field: where it is on the frame, what it says, and the
/// PII class the heuristics assigned to it.
class ReadField {
  const ReadField({
    required this.bbox,
    required this.text,
    required this.confidence,
    required this.piiClass,
  });

  final Rect bbox;
  final String text;
  final double confidence;
  final PiiClass piiClass;
}
