import 'package:flutter/material.dart';

import '../models/forecast.dart';

class EventList extends StatelessWidget {
  const EventList({super.key, required this.events});

  final List<AnomalyEvent> events;

  @override
  Widget build(BuildContext context) {
    if (events.isEmpty) {
      return const Center(
        child: Text(
          'No anomalies detected',
          style: TextStyle(color: Color(0xFF5A6A80), fontSize: 13),
        ),
      );
    }
    return ListView.builder(
      reverse: true,
      itemCount: events.length,
      itemBuilder: (context, i) {
        final e = events[events.length - 1 - i];
        final t = e.wallClock;
        final hh = t.hour.toString().padLeft(2, '0');
        final mm = t.minute.toString().padLeft(2, '0');
        final ss = t.second.toString().padLeft(2, '0');
        return ListTile(
          dense: true,
          visualDensity: VisualDensity.compact,
          leading: const Icon(Icons.warning_amber_rounded,
              color: Color(0xFFFF5470), size: 18),
          title: Text(
            'ANOMALY  value ${e.value.toStringAsFixed(2)}  '
            'score ${e.score.toStringAsFixed(2)}',
            style: const TextStyle(
                fontFamily: 'monospace', fontSize: 12, color: Colors.white),
          ),
          trailing: Text(
            '$hh:$mm:$ss  #${e.globalIndex}',
            style: const TextStyle(
                fontFamily: 'monospace',
                fontSize: 11,
                color: Color(0xFF8899AA)),
          ),
        );
      },
    );
  }
}
