import 'package:flutter/material.dart';

/// GlyphGo palette — deep "night travel" navy with a wayfinding-teal accent.
class GlyphColors {
  static const Color background = Color(0xFF0B1020);
  static const Color surface = Color(0xFF161D33);
  static const Color accent = Color(0xFF2DD4BF); // wayfinding teal
  static const Color accentAlt = Color(0xFF8B9DF8); // signpost indigo
  static const Color ok = Color(0xFF34D399);
  static const Color warn = Color(0xFFFBBF24);
}

ThemeData buildGlyphGoTheme() {
  final base = ThemeData(
    brightness: Brightness.dark,
    useMaterial3: true,
    scaffoldBackgroundColor: GlyphColors.background,
    colorScheme: const ColorScheme.dark(
      primary: GlyphColors.accent,
      secondary: GlyphColors.accentAlt,
      surface: GlyphColors.surface,
    ),
  );
  return base.copyWith(
    sliderTheme: base.sliderTheme.copyWith(
      activeTrackColor: GlyphColors.accent,
      thumbColor: GlyphColors.accent,
    ),
  );
}
