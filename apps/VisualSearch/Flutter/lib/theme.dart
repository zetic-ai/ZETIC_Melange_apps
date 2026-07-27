import 'package:flutter/material.dart';

/// SnapSeek palette: deep near-black canvas with an electric-teal accent
/// (the "scan / lock" colour) and a warm amber for the SAME-PRODUCT badge.
class SnapColors {
  const SnapColors._();

  static const Color bg = Color(0xFF0B0F14);
  static const Color surface = Color(0xFF151C24);
  static const Color accent = Color(0xFF19E3B1); // electric teal
  static const Color accentDim = Color(0xFF0E8F72);
  static const Color amber = Color(0xFFFFB020);
  static const Color textHi = Colors.white;
  static const Color textLo = Colors.white70;
}

ThemeData buildSnapTheme() {
  final base = ThemeData.dark(useMaterial3: true);
  return base.copyWith(
    scaffoldBackgroundColor: SnapColors.bg,
    colorScheme: base.colorScheme.copyWith(
      primary: SnapColors.accent,
      secondary: SnapColors.amber,
      surface: SnapColors.surface,
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        backgroundColor: SnapColors.accent,
        foregroundColor: Colors.black,
      ),
    ),
  );
}
