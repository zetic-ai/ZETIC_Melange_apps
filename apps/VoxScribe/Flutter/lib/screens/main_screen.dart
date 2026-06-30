import 'dart:async';
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';

import '../models/speaker_segment.dart';
import '../models/stage_timings.dart';
import '../models/transcript_line.dart';
import '../services/pipeline_isolate.dart';
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
  double _duration = 0;
  bool _running = false;
  String? _error;
  String _diag = ''; // live segmentation output stats (fix-verification readout)

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
    _subs.add(c.done.listen((StageTimings _) {
      if (mounted) setState(() => _running = false);
    }, onError: onErr));
    _subs.add(c.status.listen((String s) {
      if (!mounted) return;
      if (s.startsWith('ERROR:')) {
        setState(() {
          _error = s.substring(6).trim();
          _running = false;
        });
      } else if (s.startsWith('in rms=')) {
        setState(() => _diag = s); // live seg output stats (verification)
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
      _running = true;
      _error = null;
      _playPos = Duration.zero;
    });
    _playClip();
    widget.controller.runDemo(widget.wavBytes);
  }

  /// Reveal transcript + timeline progressively, synced to audio playback, so
  /// it reads like live transcription: a speaker's row appears when they start,
  /// then the words fill in WORD-BY-WORD across their segment (not all at once),
  /// and the timeline band GROWS to the current playback position.
  void _applyReveal() {
    final double pos = _playPos.inMilliseconds / 1000.0;
    final List<TranscriptLine> shown = <TranscriptLine>[];
    for (final TranscriptLine l in _allLines) {
      if (pos < l.start) continue;
      final String text;
      if (pos >= l.end || l.end <= l.start) {
        text = l.text; // finished speaking -> full line
      } else {
        // partial: reveal words proportional to elapsed time in the segment.
        final List<String> words = l.text.split(' ');
        final double frac = ((pos - l.start) / (l.end - l.start)).clamp(0.0, 1.0);
        final int k = (frac * words.length).ceil().clamp(1, words.length);
        text = words.take(k).join(' ');
      }
      shown.add(TranscriptLine(
          speaker: l.speaker, start: l.start, end: l.end, text: text));
    }
    final List<SpeakerSegment> segs = <SpeakerSegment>[];
    for (final SpeakerSegment s in _allSegments) {
      if (s.start > pos) continue;
      segs.add(SpeakerSegment(
          start: s.start, end: pos < s.end ? pos : s.end, speaker: s.speaker));
    }
    final bool grew = shown.length != _lines.length;
    setState(() {
      _lines = shown;
      _segments = segs;
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
      body: Column(
        children: <Widget>[
          const _OnDeviceBadge(),
          if (_diag.isNotEmpty)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 6),
              child: Text('seg: $_diag',
                  style: const TextStyle(
                      color: Color(0xFFFFD54F), fontSize: 10)),
            ),
          Expanded(
            child: TranscriptView(lines: _lines, scrollController: _scroll),
          ),
          if (_error != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 0, 12, 8),
              child: _ErrorBanner(message: _error!),
            ),
          // Playback progress — same left/right inset (12) as the timeline so
          // the bar and the "who spoke when" lanes line up 1:1 in time.
          _PlaybackBar(position: _playPos, duration: _playDur),
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
            child:
                TimelineWidget(segments: _segments, durationSec: _duration),
          ),
        ],
      ),
    );
  }
}

/// Slim "On-device · No cloud" header badge (replaces the floating HUD). The
/// SPEC on-device signal, kept out of the transcript's way.
class _OnDeviceBadge extends StatelessWidget {
  const _OnDeviceBadge();

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
      child: Row(
        children: <Widget>[
          const Icon(Icons.shield_outlined, size: 14, color: Color(0xFF00E5A0)),
          const SizedBox(width: 6),
          const Text('On-device · No cloud',
              style: TextStyle(
                  color: Color(0xFF00E5A0),
                  fontSize: 12,
                  fontWeight: FontWeight.w600)),
          const Spacer(),
          Text('Whisper · pyannote',
              style: TextStyle(color: Colors.white.withValues(alpha: 0.3), fontSize: 11)),
        ],
      ),
    );
  }
}

class _PlaybackBar extends StatelessWidget {
  const _PlaybackBar({required this.position, required this.duration});
  final Duration position;
  final Duration duration;

  @override
  Widget build(BuildContext context) {
    final double frac = duration.inMilliseconds > 0
        ? (position.inMilliseconds / duration.inMilliseconds).clamp(0.0, 1.0)
        : 0.0;
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 4, 12, 10),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(4),
        child: LinearProgressIndicator(
          value: frac,
          minHeight: 6,
          backgroundColor: Colors.white12,
        ),
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

