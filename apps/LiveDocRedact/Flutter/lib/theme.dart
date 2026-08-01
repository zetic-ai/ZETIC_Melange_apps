import 'package:flutter/material.dart';

import 'models/read_field.dart';

/// RedactLens palette: dark scanner UI with a teal "lens" accent.
class RedactColors {
  static const Color background = Color(0xFF0B1116);
  static const Color surface = Color(0xFF16212A);
  static const Color accent = Color(0xFF2DD4BF); // teal lens
  static const Color textBox = Color(0xFF60A5FA); // detected-text outline
  static const Color redaction = Color(0xFF0F172A); // bar fill

  static const Color name = Color(0xFFF472B6);
  static const Color dob = Color(0xFFFBBF24);
  static const Color idNumber = Color(0xFFF87171);
  static const Color mrz = Color(0xFFA78BFA);

  static Color forPii(PiiClass c) {
    switch (c) {
      case PiiClass.name:
        return name;
      case PiiClass.dob:
        return dob;
      case PiiClass.idNumber:
        return idNumber;
      case PiiClass.mrz:
        return mrz;
      case PiiClass.other:
        return textBox;
    }
  }
}

ThemeData buildRedactLensTheme() {
  return ThemeData(
    brightness: Brightness.dark,
    scaffoldBackgroundColor: RedactColors.background,
    colorScheme: const ColorScheme.dark(
      primary: RedactColors.accent,
      secondary: RedactColors.accent,
      surface: RedactColors.surface,
    ),
    sliderTheme: const SliderThemeData(
      activeTrackColor: RedactColors.accent,
      thumbColor: RedactColors.accent,
    ),
    useMaterial3: true,
  );
}
