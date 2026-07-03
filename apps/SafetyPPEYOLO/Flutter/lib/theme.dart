import 'package:flutter/material.dart';

import 'models/detection.dart';

/// SiteGuard palette — high-vis safety yellow on industrial dark.
class SiteColors {
  SiteColors._();

  /// Primary accent — high-visibility safety yellow.
  static const Color accent = Color(0xFFFFC400);

  /// PPE worn (compliant) — green family.
  static const Color hardhat = Color(0xFF2ECC71);
  static const Color vest = Color(0xFF1ABC9C);

  /// PPE missing (violation) — red family.
  static const Color noHardhat = Color(0xFFE74C3C);
  static const Color noVest = Color(0xFFFF6B35);

  static const Color background = Color(0xFF0C0E10);
  static const Color surface = Color(0xFF171A1E);

  /// Semi-transparent dark used by the HUD / stats bars.
  static const Color scrim = Color(0xCC121417);

  static Color forClass(int classId) {
    switch (classId) {
      case kClassHardhat:
        return hardhat;
      case kClassVest:
        return vest;
      case kClassNoHardhat:
        return noHardhat;
      case kClassNoVest:
        return noVest;
      default:
        return accent;
    }
  }
}

ThemeData buildSiteGuardTheme() {
  final base = ThemeData.dark(useMaterial3: true);
  return base.copyWith(
    scaffoldBackgroundColor: SiteColors.background,
    colorScheme: base.colorScheme.copyWith(
      primary: SiteColors.accent,
      secondary: SiteColors.accent,
      surface: SiteColors.surface,
    ),
    sliderTheme: base.sliderTheme.copyWith(
      activeTrackColor: SiteColors.accent,
      thumbColor: SiteColors.accent,
      overlayColor: SiteColors.accent.withValues(alpha: 0.2),
    ),
    progressIndicatorTheme: const ProgressIndicatorThemeData(
      color: SiteColors.accent,
    ),
  );
}
