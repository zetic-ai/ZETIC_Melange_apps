import 'dart:async';
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
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
  // Full pipeline output (buffered), and what's REVEALED so far. Lines and
  // segments are revealed progressively as audio playback passes each start
  // time, so a new speaker row appears when that speaker "starts speaking".
  final List<TranscriptLine> _allLines = <TranscriptLine>[];
  List<SpeakerSegment> _allSegments = <SpeakerSegment>[];
  List<TranscriptLine> _lines = <TranscriptLine>[];
  List<SpeakerSegment> _segments = <SpeakerSegment>[];
  StageTimings? _timings;
  double _duration = 0;
  bool _running = false;
  String _status = '';
  String? _error;

  final ScrollController _scroll = ScrollController();
  final List<StreamSubscription<dynamic>> _subs = <StreamSubscription<dynamic>>[];

  // Audio playback of the bundled clip (so the demo plays the conversation).
  final AudioPlayer _player = AudioPlayer();
  Duration _playPos = Duration.zero;
  Duration _playDur = Duration.zero;
  bool _playing = false;

  @override
  void initState() {
    super.initState();
    final PipelineController c = widget.controller;
    _player.onPositionChanged.listen((Duration p) {
      if (mounted) {
        _playPos = p;
        _applyReveal();
      }
    });
    _player.onDurationChanged.listen((Duration d) {
      if (mounted) setState(() => _playDur = d);
    });
    _player.onPlayerStateChanged.listen((PlayerState s) {
      if (mounted) setState(() => _playing = s == PlayerState.playing);
    });
    void onErr(Object e, StackTrace st) {
      if (mounted) {
        setState(() {
          _error = '$e';
          _running = false;
        });
      }
    }

    _subs.add(c.segments.listen((List<SpeakerSegment> s) {
      if (mounted) {
        _allSegments = s;
        _duration = c.audioDurationSec;
        _applyReveal();
      }
    }, onError: onErr));
    _subs.add(c.lines.listen((TranscriptLine line) {
      if (mounted) {
        _allLines.add(line);
        _applyReveal();
      }
    }, onError: onErr));
    _subs.add(c.done.listen((StageTimings t) {
      if (mounted) {
        setState(() {
          _timings = t;
          _running = false;
        });
      }
    }, onError: onErr));
    _subs.add(c.status.listen((String s) {
      if (mounted) {
        setState(() {
          _status = s;
          if (s.startsWith('ERROR:')) {
            _error = s.substring(6).trim();
            _running = false;
          }
        });
      }
    }));
    WidgetsBinding.instance.addPostFrameCallback((_) => _run());
  }

  void _run() {
    setState(() {
      _allLines.clear();
      _allSegments = <SpeakerSegment>[];
      _lines = <TranscriptLine>[];
      _segments = <SpeakerSegment>[];
      _timings = null;
      _running = true;
      _error = null;
      _status = 'Starting…';
      _playPos = Duration.zero;
    });
    _playClip();
    widget.controller.runDemo(widget.wavBytes);
  }

  /// Reveal lines/segments whose start time has been reached by audio playback,
  /// so each speaker's row appears as they "start speaking" (synced to audio).
  void _applyReveal() {
    final double posSec = _playPos.inMilliseconds / 1000.0;
    final List<TranscriptLine> revealed = _allLines
        .where((TranscriptLine l) => l.start <= posSec)
        .toList(growable: false);
    final bool grew = revealed.length != _lines.length;
    setState(() {
      _lines = revealed;
      _segments = _allSegments
          .where((SpeakerSegment s) => s.start <= posSec)
          .toList(growable: false);
    });
    if (grew) _autoScroll();
  }

  Future<void> _playClip() async {
    try {
      await _player.stop();
      // AssetSource resolves under assets/ (the clip is declared in pubspec).
      await _player.play(AssetSource('demo_2spk.wav'));
    } catch (_) {
      // Playback is a demo nicety; never let it block the pipeline.
    }
  }

  Future<void> _togglePlay() async {
    if (_playing) {
      await _player.pause();
    } else {
      await _player.resume();
    }
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
    _player.dispose();
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
            tooltip: _playing ? 'Pause audio' : 'Play audio',
            onPressed: _togglePlay,
            icon: Icon(_playing ? Icons.pause : Icons.play_arrow),
          ),
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
              _PlaybackBar(position: _playPos, duration: _playDur),
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
            Positioned(
              left: 12,
              top: 12,
              child: _RunningChip(status: _status),
            ),
          if (_error != null)
            Positioned(
              left: 12,
              right: 12,
              bottom: 12,
              child: _ErrorBanner(message: _error!),
            ),
        ],
      ),
    );
  }
}

class _PlaybackBar extends StatelessWidget {
  const _PlaybackBar({required this.position, required this.duration});
  final Duration position;
  final Duration duration;

  String _fmt(Duration d) =>
      '${d.inSeconds}.${(d.inMilliseconds % 1000) ~/ 100}s';

  @override
  Widget build(BuildContext context) {
    final double frac = duration.inMilliseconds > 0
        ? (position.inMilliseconds / duration.inMilliseconds).clamp(0.0, 1.0)
        : 0.0;
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 4, 12, 8),
      child: Row(
        children: <Widget>[
          const Icon(Icons.graphic_eq, size: 16, color: Colors.white38),
          const SizedBox(width: 8),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: frac,
                minHeight: 4,
                backgroundColor: Colors.white12,
              ),
            ),
          ),
          const SizedBox(width: 8),
          Text('${_fmt(position)} / ${_fmt(duration)}',
              style: const TextStyle(color: Colors.white38, fontSize: 11)),
        ],
      ),
    );
  }
}

class _ErrorBanner extends StatelessWidget {
  const _ErrorBanner({required this.message});
  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF7F1D1D).withValues(alpha: 0.95),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          const Icon(Icons.error_outline, color: Colors.white, size: 18),
          const SizedBox(width: 8),
          Expanded(
            child: Text(message,
                style: const TextStyle(color: Colors.white, fontSize: 12)),
          ),
        ],
      ),
    );
  }
}

class _RunningChip extends StatelessWidget {
  const _RunningChip({required this.status});
  final String status;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.55),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          const SizedBox(
              width: 12,
              height: 12,
              child: CircularProgressIndicator(strokeWidth: 2)),
          const SizedBox(width: 8),
          Text(status.isEmpty ? 'Transcribing…' : status,
              style: const TextStyle(color: Colors.white70, fontSize: 12)),
        ],
      ),
    );
  }
}
