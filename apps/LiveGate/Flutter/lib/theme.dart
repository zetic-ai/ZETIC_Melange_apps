import 'package:flutter/material.dart';

/// TrueFace brand palette. Trust-coded (teal/cyan), dark-first, with clear
/// pass/fail semantics.
class GateColors {
  GateColors._();

  /// Primary accent — cyan/teal, "verified".
  static const Color accent = Color(0xFF00E5C7);

  /// PASS / LIVE.
  static const Color pass = Color(0xFF23D18B);

  /// SPOOF / FAIL.
  static const Color fail = Color(0xFFFF4D5E);

  /// Neutral warning (live but no match / no reference).
  static const Color warn = Color(0xFFFFC24B);

  static const Color background = Color(0xFF07090C);
  static const Color surface = Color(0xFF13161B);

  /// Semi-transparent dark used by the HUD / verdict bars.
  static const Color scrim = Color(0xCC0C0E12);
}

ThemeData buildGateTheme() {
  final base = ThemeData.dark(useMaterial3: true);
  return base.copyWith(
    scaffoldBackgroundColor: GateColors.background,
    colorScheme: base.colorScheme.copyWith(
      primary: GateColors.accent,
      secondary: GateColors.accent,
      surface: GateColors.surface,
    ),
    progressIndicatorTheme: const ProgressIndicatorThemeData(
      color: GateColors.accent,
    ),
  );
}
