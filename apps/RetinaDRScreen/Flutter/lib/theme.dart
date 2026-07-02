import 'package:flutter/material.dart';

/// FundusGate visual theme — a clean, clinical dark palette.
class FundusTheme {
  const FundusTheme._();

  static const Color background = Color(0xFF0B1418);
  static const Color surface = Color(0xFF14232B);
  static const Color surfaceAlt = Color(0xFF1B2E38);
  static const Color primary = Color(0xFF1FB6B0); // clinical teal
  static const Color onSurfaceMuted = Color(0xFF9BB4BD);

  /// Verdict colors.
  static const Color referable = Color(0xFFE4572E); // amber-red alert
  static const Color notReferable = Color(0xFF3DBE8B); // reassuring green

  static ThemeData build() {
    final base = ThemeData.dark(useMaterial3: true);
    return base.copyWith(
      scaffoldBackgroundColor: background,
      colorScheme: base.colorScheme.copyWith(
        primary: primary,
        surface: surface,
        brightness: Brightness.dark,
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: background,
        elevation: 0,
        centerTitle: true,
      ),
      cardTheme: const CardThemeData(
        color: surface,
        elevation: 0,
        margin: EdgeInsets.zero,
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: primary,
          foregroundColor: Colors.black,
          padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 20),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
        ),
      ),
    );
  }
}
