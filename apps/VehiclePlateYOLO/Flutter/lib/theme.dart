import 'package:flutter/material.dart';

/// Trade-show visual language: dark, high-contrast, a single electric accent.
class AppTheme {
  static const Color bg = Color(0xFF0B0F14);
  static const Color surface = Color(0xFF141B24);
  static const Color accent = Color(0xFF18E0C8); // plate-scan cyan
  static const Color accentSoft = Color(0x3318E0C8);
  static const Color warn = Color(0xFFFFB020);
  static const Color textPrimary = Color(0xFFEAF2F6);
  static const Color textMuted = Color(0xFF8A99A8);

  static ThemeData get dark => ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    scaffoldBackgroundColor: bg,
    colorScheme: const ColorScheme.dark(
      primary: accent,
      surface: surface,
      onPrimary: bg,
    ),
    fontFamily: 'SF Pro',
  );
}
