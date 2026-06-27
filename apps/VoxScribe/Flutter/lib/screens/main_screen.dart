import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../models/speaker_segment.dart';
import '../models/stage_timings.dart';
import '../models/transcript_line.dart';
import '../services/pipeline_isolate.dart';
import '../widgets/hud.dart';
import '../widgets/timeline_widget.dart';
import '../widgets/transcript_view.dart';

/// The demo screen: runs the bundled clip through the on-device pipeline and
/// progressively paints the speaker-labeled transcript + who-spoke-when
/// timeline + the on-device/RTF HUD.
class MainScreen extends StatefulWidget {
  const MainScreen({
    super.key,
    required this.controller,
    required this.wavBytes,
  });

  final PipelineController controller;
  final Uint8List wavBytes;

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  final List<TranscriptLine> _lines = <TranscriptLine>[];
  List<SpeakerSegment> _segments = <SpeakerSegment>[];
  StageTimings? _timings;
  double _duration = 0;
  bool _running = false;

  final ScrollController _scroll = ScrollController();
  final List<StreamSubscription<dynamic>> _subs = <StreamSubscription<dynamic>>[];

  @override
  void initState() {
    super.initState();
    final PipelineController c = widget.controller;
    _subs.add(c.segments.listen((List<SpeakerSegment> s) {
      if (mounted) {
        setState(() {
          _segments = s;
          _duration = c.audioDurationSec;
        });
      }
    }));
    _subs.add(c.lines.listen((TranscriptLine line) {
      if (mounted) {
        setState(() => _lines.add(line));
        _autoScroll();
      }
    }));
    _subs.add(c.done.listen((StageTimings t) {
      if (mounted) {
        setState(() {
          _timings = t;
          _running = false;
        });
      }
    }));
    WidgetsBinding.instance.addPostFrameCallback((_) => _run());
  }

  void _run() {
    setState(() {
      _lines.clear();
      _segments = <SpeakerSegment>[];
      _timings = null;
      _running = true;
    });
    widget.controller.runDemo(widget.wavBytes);
  }

  void _autoScroll() {
    if (_scroll.hasClients) {
      _scroll.animateTo(
        _scroll.position.maxScrollExtent + 80,
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeOut,
      );
    }
  }

  @override
  void dispose() {
    for (final StreamSubscription<dynamic> s in _subs) {
      s.cancel();
    }
    _scroll.dispose();
    widget.controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('VoxScribe'),
        backgroundColor: Colors.transparent,
        actions: <Widget>[
          IconButton(
            tooltip: 'Re-run demo clip',
            onPressed: _running ? null : _run,
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: Stack(
        children: <Widget>[
          Column(
            children: <Widget>[
              Expanded(
                child: TranscriptView(lines: _lines, scrollController: _scroll),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
                child: TimelineWidget(
                    segments: _segments, durationSec: _duration),
              ),
            ],
          ),
          Positioned(
            right: 12,
            top: 12,
            child: Hud(
              timings: _timings,
              servedNote: 'Served backend = device truth; read runtimeApType '
                  'on the console.',
            ),
          ),
          if (_running)
            const Positioned(
              left: 12,
              top: 12,
              child: _RunningChip(),
            ),
        ],
      ),
    );
  }
}

class _RunningChip extends StatelessWidget {
  const _RunningChip();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.55),
        borderRadius: BorderRadius.circular(10),
      ),
      child: const Row(
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          SizedBox(
              width: 12,
              height: 12,
              child: CircularProgressIndicator(strokeWidth: 2)),
          SizedBox(width: 8),
          Text('Transcribing…',
              style: TextStyle(color: Colors.white70, fontSize: 12)),
        ],
      ),
    );
  }
}
