import 'package:flutter/material.dart';

import '../models/stage_timings.dart';

/// Speaker color palette (1-based speaker label -> color).
const List<Color> kSpeakerColors = <Color>[
  Color(0xFF00C2A8), // Speaker 1 — teal
  Color(0xFFFF8A65), // Speaker 2 — orange
  Color(0xFFB39DDB), // Speaker 3 — purple
];

Color speakerColor(int speaker1Based) =>
    kSpeakerColors[(speaker1Based - 1) % kSpeakerColors.length];

/// The on-device signal + diagnostics HUD (SPEC must-have; CLAUDE.md §5 — these
/// live on screen because Dart logs do not surface in a release device console).
///
/// "On-device · No cloud" is a STATIC badge (GATE-2 decision 10): after the
/// one-time model download the app makes no network calls. Per-stage latencies
/// and RTF are shown so the human can read device-real numbers off-screen.
class Hud extends StatelessWidget {
  const Hud({super.key, required this.timings, required this.servedNote});

  final StageTimings? timings;

  /// Served-artifact reminder line (the truth is on the native console).
  final String servedNote;

  @override
  Widget build(BuildContext context) {
    final StageTimings? t = timings;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.55),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          Row(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              const Icon(Icons.shield_outlined,
                  size: 14, color: Color(0xFF00E5A0)),
              const SizedBox(width: 6),
              Text('On-device · No cloud',
                  style: const TextStyle(
                      color: Color(0xFF00E5A0),
                      fontSize: 12,
                      fontWeight: FontWeight.w600)),
            ],
          ),
          const SizedBox(height: 6),
          if (t == null)
            const Text('Running pipeline…',
                style: TextStyle(color: Colors.white54, fontSize: 11))
          else ...<Widget>[
            _row('RTF', '${t.rtf.toStringAsFixed(2)}×  '
                '(${t.totalMs.toStringAsFixed(0)} ms / '
                '${t.audioDurationSec.toStringAsFixed(1)} s)'),
            _row('seg run', '${t.segRunMs.toStringAsFixed(0)} ms'),
            _row('log-mel', '${t.logMelMs.toStringAsFixed(0)} ms (Dart)'),
            _row('enc run', '${t.encRunMs.toStringAsFixed(0)} ms'),
            _row('dec run', '${t.decRunMs.toStringAsFixed(0)} ms'),
            _row('powerset', '${t.powersetMs.toStringAsFixed(1)} ms (Dart)'),
            _row('segments', '${t.segmentsFound}'),
            if (t.diag.isNotEmpty) ...<Widget>[
              const SizedBox(height: 4),
              Text(t.diag,
                  style: const TextStyle(
                      color: Color(0xFFFFD54F),
                      fontSize: 10,
                      fontFeatures: <FontFeature>[
                        FontFeature.tabularFigures()
                      ])),
            ],
          ],
          const SizedBox(height: 6),
          Text(servedNote,
              style: const TextStyle(color: Colors.white30, fontSize: 10)),
        ],
      ),
    );
  }

  Widget _row(String k, String v) => Padding(
        padding: const EdgeInsets.only(top: 2),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            SizedBox(
                width: 64,
                child: Text(k,
                    style:
                        const TextStyle(color: Colors.white38, fontSize: 11))),
            Text(v,
                style: const TextStyle(
                    color: Colors.white70,
                    fontSize: 11,
                    fontFeatures: <FontFeature>[FontFeature.tabularFigures()])),
          ],
        ),
      );
}
