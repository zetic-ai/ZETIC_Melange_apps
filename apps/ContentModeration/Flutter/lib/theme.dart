import 'package:flutter/material.dart';

/// SafeLens visual identity — a calm, trust-and-safety dark theme with the three
/// decision-band colors (green / amber / red) carried by [Decision] itself.
class SafeLensTheme {
  const SafeLensTheme._();

  static const Color background = Color(0xFF0E1116);
  static const Color surface = Color(0xFF1A1F27);
  static const Color surfaceMuted = Color(0xFF232A34);
  static const Color onSurface = Color(0xFFE7ECF3);
  static const Color onSurfaceMuted = Color(0xFF8A94A3);
  static const Color accent = Color(0xFF3D7EFF);

  static ThemeData build() {
    const scheme = ColorScheme.dark(
      primary: accent,
      surface: surface,
      onSurface: onSurface,
    );
    return ThemeData(
      useMaterial3: true,
      colorScheme: scheme,
      scaffoldBackgroundColor: background,
      appBarTheme: const AppBarTheme(
        backgroundColor: background,
        foregroundColor: onSurface,
        centerTitle: true,
        elevation: 0,
      ),
      cardTheme: const CardThemeData(
        color: surface,
        elevation: 0,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.all(Radius.circular(16)),
        ),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: accent,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
      ),
    );
  }
}
