import 'package:flutter/material.dart';

/// SayRight brand palette. Calm, coaching, dark-first.
class SayColors {
  SayColors._();

  /// Primary accent — indigo/violet.
  static const Color accent = Color(0xFF7C6CF0);
  static const Color accentSoft = Color(0xFF9C8FF6);

  static const Color background = Color(0xFF0C0C12);
  static const Color surface = Color(0xFF17171F);
  static const Color scrim = Color(0xCC13131A);

  // Score bands (calibrated 0..100).
  static const Color great = Color(0xFF35C48B); // >= 80
  static const Color good = Color(0xFF8FD14F); // 60–79
  static const Color okay = Color(0xFFF2B33D); // 40–59
  static const Color weak = Color(0xFFF06C6C); // < 40

  /// Map a calibrated 0..100 score to its band color.
  static Color forScore(double score) {
    if (score >= 80) return great;
    if (score >= 60) return good;
    if (score >= 40) return okay;
    return weak;
  }
}

ThemeData buildSayRightTheme() {
  final base = ThemeData.dark(useMaterial3: true);
  return base.copyWith(
    scaffoldBackgroundColor: SayColors.background,
    colorScheme: base.colorScheme.copyWith(
      primary: SayColors.accent,
      secondary: SayColors.accentSoft,
      surface: SayColors.surface,
    ),
    progressIndicatorTheme: const ProgressIndicatorThemeData(
      color: SayColors.accent,
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(backgroundColor: SayColors.accent),
    ),
  );
}
