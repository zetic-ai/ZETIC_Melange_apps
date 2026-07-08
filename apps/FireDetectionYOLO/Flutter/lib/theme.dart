import 'package:flutter/material.dart';

/// PyroGuard brand palette. Fire-coded, dark-first.
class PyroColors {
  PyroColors._();

  /// Primary accent — orange-red, "fire".
  static const Color accent = Color(0xFFFF4500);
  static const Color fire = Color(0xFFFF4500);
  static const Color smoke = Color(0xFF9E9E9E);

  static const Color background = Color(0xFF0B0B0D);
  static const Color surface = Color(0xFF16161A);

  /// Semi-transparent dark used by the HUD / stats bars.
  static const Color scrim = Color(0xCC121214);
}

ThemeData buildPyroGuardTheme() {
  final base = ThemeData.dark(useMaterial3: true);
  return base.copyWith(
    scaffoldBackgroundColor: PyroColors.background,
    colorScheme: base.colorScheme.copyWith(
      primary: PyroColors.accent,
      secondary: PyroColors.accent,
      surface: PyroColors.surface,
    ),
    sliderTheme: base.sliderTheme.copyWith(
      activeTrackColor: PyroColors.accent,
      thumbColor: PyroColors.accent,
      overlayColor: PyroColors.accent.withValues(alpha: 0.2),
    ),
    progressIndicatorTheme: const ProgressIndicatorThemeData(
      color: PyroColors.accent,
    ),
  );
}
