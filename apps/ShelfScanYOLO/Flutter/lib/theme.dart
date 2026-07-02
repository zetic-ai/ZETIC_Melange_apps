import 'package:flutter/material.dart';

/// ShelfSense visual identity — a dark, retail-analytics HUD palette with a
/// bright "scan green" accent that reads well over shelf photography.
class ShelfSenseTheme {
  static const Color accent = Color(0xFF13E27B); // scan green (box + count)
  static const Color accentDim = Color(0xFF0AAE5E);
  static const Color background = Color(0xFF0B0F14);
  static const Color surface = Color(0xFF151C24);
  static const Color surfaceHigh = Color(0xFF1E2833);

  static ThemeData get dark {
    final base = ThemeData.dark(useMaterial3: true);
    return base.copyWith(
      scaffoldBackgroundColor: background,
      colorScheme: base.colorScheme.copyWith(
        primary: accent,
        secondary: accent,
        surface: surface,
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: background,
        elevation: 0,
        centerTitle: false,
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: accent,
          foregroundColor: Colors.black,
          textStyle: const TextStyle(fontWeight: FontWeight.w700),
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        ),
      ),
    );
  }
}
