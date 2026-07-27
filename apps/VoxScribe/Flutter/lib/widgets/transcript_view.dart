import 'package:flutter/material.dart';

import '../models/transcript_line.dart';
import 'hud.dart';

/// Scrolling, speaker-colored transcript. New lines append progressively as the
/// pipeline finishes each span (R4 progressive render).
class TranscriptView extends StatelessWidget {
  const TranscriptView({super.key, required this.lines, this.scrollController});

  final List<TranscriptLine> lines;
  final ScrollController? scrollController;

  @override
  Widget build(BuildContext context) {
    if (lines.isEmpty) {
      return const Center(
        child: Text('Listening…',
            style: TextStyle(color: Colors.white38, fontSize: 14)),
      );
    }
    return ListView.builder(
      controller: scrollController,
      padding: const EdgeInsets.all(12),
      itemCount: lines.length,
      itemBuilder: (BuildContext context, int i) {
        final TranscriptLine line = lines[i];
        final Color c = speakerColor(line.speaker);
        return Padding(
          padding: const EdgeInsets.only(bottom: 12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              Row(
                children: <Widget>[
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 8, vertical: 2),
                    decoration: BoxDecoration(
                      color: c.withValues(alpha: 0.18),
                      borderRadius: BorderRadius.circular(6),
                      border: Border.all(color: c.withValues(alpha: 0.6)),
                    ),
                    child: Text('Speaker ${line.speaker}',
                        style: TextStyle(
                            color: c,
                            fontSize: 12,
                            fontWeight: FontWeight.w600)),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    '${line.start.toStringAsFixed(1)}–${line.end.toStringAsFixed(1)}s',
                    style:
                        const TextStyle(color: Colors.white30, fontSize: 11),
                  ),
                ],
              ),
              const SizedBox(height: 4),
              Text(line.text,
                  style: const TextStyle(
                      color: Colors.white, fontSize: 16, height: 1.3)),
            ],
          ),
        );
      },
    );
  }
}
