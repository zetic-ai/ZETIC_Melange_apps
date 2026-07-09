import 'package:flutter/material.dart';

/// Clinical-modern palette: deep slate canvas, medical teal accent, coral for the
/// LV overlay, restrained greens/ambers for status.
class Palette {
  static const bg0 = Color(0xFF0A0E14);
  static const bg1 = Color(0xFF0E141C);
  static const card = Color(0xFF121A24);
  static const cardHi = Color(0xFF17212E);
  static const stroke = Color(0x14FFFFFF); // white @ 8%
  static const strokeHi = Color(0x24FFFFFF);

  static const teal = Color(0xFF2DD4BF);
  static const cyan = Color(0xFF38BDF8);
  static const coral = Color(0xFFFF5A5F); // LV mask tint
  static const green = Color(0xFF34D399);
  static const amber = Color(0xFFFBBF24);

  static const textHi = Color(0xFFF1F5F9);
  static const textMid = Color(0xFF94A3B8);
  static const textLow = Color(0xFF5B6675);
}

ThemeData buildTheme() {
  final base = ThemeData.dark(useMaterial3: true);
  return base.copyWith(
    scaffoldBackgroundColor: Palette.bg0,
    colorScheme: base.colorScheme.copyWith(
      primary: Palette.teal,
      surface: Palette.card,
    ),
    textTheme: base.textTheme.apply(
      bodyColor: Palette.textHi,
      displayColor: Palette.textHi,
    ),
  );
}

const _mono = 'SF Mono';

TextStyle numStyle({
  double size = 20,
  Color color = Palette.textHi,
  FontWeight w = FontWeight.w600,
}) => TextStyle(
  fontFamily: _mono,
  fontFamilyFallback: const ['Menlo', 'RobotoMono', 'monospace'],
  fontSize: size,
  fontWeight: w,
  color: color,
  letterSpacing: -0.5,
  fontFeatures: const [FontFeature.tabularFigures()],
);

TextStyle labelStyle({Color color = Palette.textMid}) => TextStyle(
  fontSize: 10.5,
  fontWeight: FontWeight.w600,
  color: color,
  letterSpacing: 1.1,
);

/// A soft surface card with a hairline border.
class Panel extends StatelessWidget {
  const Panel({
    super.key,
    required this.child,
    this.padding,
    this.radius = 16,
    this.color,
  });
  final Widget child;
  final EdgeInsets? padding;
  final double radius;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: padding,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [color ?? Palette.cardHi, color ?? Palette.card],
        ),
        borderRadius: BorderRadius.circular(radius),
        border: Border.all(color: Palette.stroke),
      ),
      child: child,
    );
  }
}

/// A single labeled metric with a FIXED width, so changing values never reflow
/// neighbouring tiles. Sub line is always reserved (fixed height) for the same reason.
class StatTile extends StatelessWidget {
  const StatTile({
    super.key,
    required this.label,
    required this.value,
    this.sub,
    this.accent = Palette.textHi,
    this.icon,
    this.width,
  });
  final String label;
  final String value;
  final String? sub;
  final Color accent;
  final IconData? icon;
  final double? width;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: width,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              if (icon != null) ...[
                Icon(icon, size: 12, color: Palette.textLow),
                const SizedBox(width: 5),
              ],
              Expanded(
                child: Text(
                  label.toUpperCase(),
                  style: labelStyle(),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
          const SizedBox(height: 5),
          Text(value, style: numStyle(size: 18, color: accent), maxLines: 1),
          const SizedBox(height: 1),
          SizedBox(
            height: 13,
            child: Text(
              sub ?? '',
              style: numStyle(
                size: 10.5,
                color: Palette.textLow,
                w: FontWeight.w500,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// A small pill (status badge).
class Pill extends StatelessWidget {
  const Pill({
    super.key,
    required this.child,
    this.color = Palette.textMid,
    this.dot = false,
  });
  final Widget child;
  final Color color;
  final bool dot;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.10),
        borderRadius: BorderRadius.circular(30),
        border: Border.all(color: color.withValues(alpha: 0.30)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (dot) ...[
            Container(
              width: 6,
              height: 6,
              decoration: BoxDecoration(color: color, shape: BoxShape.circle),
            ),
            const SizedBox(width: 6),
          ],
          DefaultTextStyle.merge(
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              color: color,
            ),
            child: child,
          ),
        ],
      ),
    );
  }
}
