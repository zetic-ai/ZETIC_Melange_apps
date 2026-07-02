import 'package:flutter/material.dart';

/// GradeVue visual theme — a clean, clinical dark palette with a 5-grade
/// severity color scale (green → amber → red as DR severity rises).
class GradeVueTheme {
  const GradeVueTheme._();

  static const Color background = Color(0xFF0B1418);
  static const Color surface = Color(0xFF14232B);
  static const Color surfaceAlt = Color(0xFF1B2E38);
  static const Color primary = Color(0xFF1FB6B0); // clinical teal
  static const Color onSurfaceMuted = Color(0xFF9BB4BD);

  /// Referable flag colors (grade >= 2).
  static const Color referable = Color(0xFFE4572E); // amber-red alert
  static const Color notReferable = Color(0xFF3DBE8B); // reassuring green

  /// Per-grade severity colors, index == grade 0..4. Grades < 2 are the
  /// reassuring (not-referable) hues; grades >= 2 escalate through amber to red.
  static const List<Color> gradeColors = [
    Color(0xFF3DBE8B), // 0 No DR — green
    Color(0xFF8FCf6A), // 1 Mild — yellow-green
    Color(0xFFE8B23A), // 2 Moderate — amber
    Color(0xFFE4722E), // 3 Severe — orange
    Color(0xFFE4572E), // 4 Proliferative — red
  ];

  static Color gradeColor(int grade) =>
      gradeColors[grade.clamp(0, gradeColors.length - 1)];

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
