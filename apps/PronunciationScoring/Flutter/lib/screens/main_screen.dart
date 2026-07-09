import 'package:flutter/material.dart';

import '../models/scoring.dart';
import '../models/sentence.dart';
import '../services/audio_recorder.dart';
import '../services/melange_service.dart';
import '../theme.dart';
import '../widgets/hud.dart';
import '../widgets/record_ring.dart';
import '../widgets/score_view.dart';

enum _Phase { idle, recording, scoring, done }

class MainScreen extends StatefulWidget {
  const MainScreen({
    super.key,
    required this.service,
    required this.sentences,
  });

  final MelangeService service;
  final List<PracticeSentence> sentences;

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  final AudioCaptureService _capture = AudioCaptureService();

  int _sentenceIndex = 0;
  _Phase _phase = _Phase.idle;
  double _progress = 0;
  PronunciationResult? _result;
  String? _error;

  PracticeSentence get _sentence => widget.sentences[_sentenceIndex];

  @override
  void dispose() {
    _capture.dispose();
    super.dispose();
  }

  Future<void> _start() async {
    if (!await _capture.hasPermission()) {
      setState(() => _error = 'Microphone permission is required to score.');
      return;
    }
    setState(() {
      _error = null;
      _result = null;
      _phase = _Phase.recording;
      _progress = 0;
    });
    try {
      final capture = await _capture.captureWindow(
        onProgress: (p) {
          if (mounted && _phase == _Phase.recording) {
            setState(() => _progress = p);
          }
        },
      );
      if (!mounted) return;
      if (capture == null) {
        // Cancelled — discard, back to idle. Never score a partial window.
        setState(() => _phase = _Phase.idle);
        return;
      }
      setState(() => _phase = _Phase.scoring);
      // One inference + sub-ms scoring; runs on the main isolate by design.
      final result =
          widget.service.score(capture.pcmBytes, capture.rate, _sentence);
      if (!mounted) return;
      setState(() {
        _result = result;
        _phase = _Phase.done;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = '$e';
        _phase = _Phase.idle;
      });
    }
  }

  void _cancel() {
    _capture.cancel();
  }

  void _nextSentence() {
    setState(() {
      _sentenceIndex = (_sentenceIndex + 1) % widget.sentences.length;
      _result = null;
      _error = null;
      _phase = _Phase.idle;
      _progress = 0;
    });
  }

  @override
  Widget build(BuildContext context) {
    final r = _result;
    return Scaffold(
      body: Column(
        children: [
          Hud(
            inferenceMs: r?.inferenceMs ?? 0,
            scoringMs: r?.scoringMs ?? 0,
            servedArtifact: widget.service.servedArtifact,
            sampleRateInfo: r?.sampleRateInfo,
          ),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(20, 18, 20, 28),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _SentenceCard(
                    sentence: _sentence,
                    onSkip: _nextSentence,
                  ),
                  const SizedBox(height: 24),
                  if (_error != null) _ErrorNote(message: _error!),
                  if (r != null) ...[
                    ScoreView(result: r),
                    const SizedBox(height: 20),
                  ],
                  Center(
                    child: Column(
                      children: [
                        RecordRing(
                          progress: _progress,
                          recording: _phase == _Phase.recording,
                          scoring: _phase == _Phase.scoring,
                          onStart: _start,
                          onCancel: _cancel,
                        ),
                        const SizedBox(height: 12),
                        Text(
                          _hint(),
                          style: TextStyle(
                            color: Colors.white.withValues(alpha: 0.6),
                            fontSize: 13,
                          ),
                        ),
                        if (r != null) ...[
                          const SizedBox(height: 16),
                          TextButton.icon(
                            onPressed: _nextSentence,
                            icon: const Icon(Icons.arrow_forward_rounded),
                            label: const Text('Next sentence'),
                          ),
                        ],
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  String _hint() {
    switch (_phase) {
      case _Phase.idle:
        return _result == null
            ? 'Tap to read the sentence aloud'
            : 'Tap to try again';
      case _Phase.recording:
        return 'Listening… tap to cancel';
      case _Phase.scoring:
        return 'Scoring…';
      case _Phase.done:
        return 'Tap to try again';
    }
  }
}

class _SentenceCard extends StatelessWidget {
  const _SentenceCard({required this.sentence, required this.onSkip});
  final PracticeSentence sentence;
  final VoidCallback onSkip;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: SayColors.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withValues(alpha: 0.06)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.menu_book_rounded,
                  color: SayColors.accentSoft, size: 18),
              const SizedBox(width: 8),
              Text('Read this aloud',
                  style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.6),
                      fontSize: 13)),
              const Spacer(),
              InkWell(
                onTap: onSkip,
                borderRadius: BorderRadius.circular(20),
                child: const Padding(
                  padding: EdgeInsets.all(4),
                  child: Icon(Icons.shuffle_rounded,
                      color: Colors.white54, size: 18),
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Text(
            sentence.text,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 24,
              fontWeight: FontWeight.w700,
              height: 1.35,
            ),
          ),
          const SizedBox(height: 10),
          Text(
            '~${sentence.estSeconds.toStringAsFixed(1)} s • ${sentence.phonemeIds.length} sounds',
            style: TextStyle(
                color: Colors.white.withValues(alpha: 0.4), fontSize: 12),
          ),
        ],
      ),
    );
  }
}

class _ErrorNote extends StatelessWidget {
  const _ErrorNote({required this.message});
  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 20),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: SayColors.weak.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: SayColors.weak.withValues(alpha: 0.5)),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_outline, color: SayColors.weak, size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: Text(message,
                style: const TextStyle(color: Colors.white70, fontSize: 13)),
          ),
        ],
      ),
    );
  }
}
