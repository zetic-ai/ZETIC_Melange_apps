import 'package:flutter/material.dart';

import '../models/read_field.dart';
import '../theme.dart';

/// Bottom bar: live per-class PII counts + fields-read/tracked tally.
class StatsBar extends StatelessWidget {
  const StatsBar({
    super.key,
    required this.piiCounts,
    required this.readCount,
    required this.trackedCount,
  });

  final Map<PiiClass, int> piiCounts;
  final int readCount;
  final int trackedCount;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      top: false,
      child: Container(
        margin: const EdgeInsets.fromLTRB(12, 0, 12, 10),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.55),
          borderRadius: BorderRadius.circular(14),
        ),
        child: Row(
          children: [
            for (final cls in const [
              PiiClass.name,
              PiiClass.dob,
              PiiClass.idNumber,
              PiiClass.mrz,
            ]) ...[
              _PiiChip(cls: cls, count: piiCounts[cls] ?? 0),
              const SizedBox(width: 8),
            ],
            const Spacer(),
            Text(
              'read $readCount/$trackedCount',
              style: const TextStyle(
                color: Colors.white70,
                fontSize: 11,
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PiiChip extends StatelessWidget {
  const _PiiChip({required this.cls, required this.count});

  final PiiClass cls;
  final int count;

  @override
  Widget build(BuildContext context) {
    final color = RedactColors.forPii(cls);
    final bool active = count > 0;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
      decoration: BoxDecoration(
        color: active
            ? color.withValues(alpha: 0.22)
            : Colors.white.withValues(alpha: 0.06),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(
          color: active ? color : Colors.white24,
          width: 1,
        ),
      ),
      child: Text(
        '${cls.label} $count',
        style: TextStyle(
          color: active ? color : Colors.white38,
          fontSize: 10.5,
          fontWeight: FontWeight.w800,
          letterSpacing: 0.4,
          fontFeatures: const [FontFeature.tabularFigures()],
        ),
      ),
    );
  }
}
