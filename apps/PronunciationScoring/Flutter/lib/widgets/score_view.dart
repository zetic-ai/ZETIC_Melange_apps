import 'package:flutter/material.dart';

import '../models/scoring.dart';
import '../theme.dart';

/// Results view: overall score, per-word color-coded chips (tap a word to see
/// its per-phoneme breakdown with the weakest sound highlighted), and a
/// collapsible "what we heard" greedy decode (decoration only).
class ScoreView extends StatefulWidget {
  const ScoreView({super.key, required this.result});

  final PronunciationResult result;

  @override
  State<ScoreView> createState() => _ScoreViewState();
}

class _ScoreViewState extends State<ScoreView> {
  int _expandedWord = -1;

  @override
  Widget build(BuildContext context) {
    final r = widget.result;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _OverallBadge(score: r.overallScore),
        const SizedBox(height: 16),
        Text('Tap a word to see each sound',
            style: TextStyle(
                color: Colors.white.withValues(alpha: 0.5), fontSize: 12)),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            for (var i = 0; i < r.words.length; i++)
              _WordChip(
                word: r.words[i],
                selected: _expandedWord == i,
                onTap: () => setState(
                    () => _expandedWord = _expandedWord == i ? -1 : i),
              ),
          ],
        ),
        if (_expandedWord >= 0) ...[
          const SizedBox(height: 14),
          _PhonemeBreakdown(word: r.words[_expandedWord]),
        ],
        const SizedBox(height: 18),
        _WhatWeHeard(phonemes: r.greedyPhonemes),
      ],
    );
  }
}

class _OverallBadge extends StatelessWidget {
  const _OverallBadge({required this.score});
  final double score;

  @override
  Widget build(BuildContext context) {
    final color = SayColors.forScore(score);
    return Row(
      children: [
        Container(
          width: 84,
          height: 84,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: color.withValues(alpha: 0.18),
            border: Border.all(color: color, width: 3),
          ),
          alignment: Alignment.center,
          child: Text(
            score.round().toString(),
            style: TextStyle(
                color: color, fontSize: 30, fontWeight: FontWeight.w900),
          ),
        ),
        const SizedBox(width: 16),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Overall',
                style: TextStyle(color: Colors.white70, fontSize: 13)),
            const SizedBox(height: 2),
            Text(
              _label(score),
              style: TextStyle(
                  color: color, fontSize: 20, fontWeight: FontWeight.w800),
            ),
          ],
        ),
      ],
    );
  }

  static String _label(double s) {
    if (s >= 80) return 'Great!';
    if (s >= 60) return 'Good';
    if (s >= 40) return 'Keep practicing';
    return 'Try again';
  }
}

class _WordChip extends StatelessWidget {
  const _WordChip(
      {required this.word, required this.selected, required this.onTap});
  final WordScore word;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final color = SayColors.forScore(word.score);
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: color.withValues(alpha: selected ? 0.35 : 0.16),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
              color: color.withValues(alpha: selected ? 1 : 0.5), width: 1.5),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(word.word,
                style: const TextStyle(
                    color: Colors.white,
                    fontSize: 15,
                    fontWeight: FontWeight.w600)),
            const SizedBox(width: 6),
            Text('${word.score.round()}',
                style: TextStyle(
                    color: color,
                    fontSize: 13,
                    fontWeight: FontWeight.w800)),
          ],
        ),
      ),
    );
  }
}

class _PhonemeBreakdown extends StatelessWidget {
  const _PhonemeBreakdown({required this.word});
  final WordScore word;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: SayColors.surface,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('“${word.word}” sounds',
              style: const TextStyle(
                  color: Colors.white,
                  fontSize: 15,
                  fontWeight: FontWeight.w700)),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              for (var i = 0; i < word.phonemes.length; i++)
                _PhonemePill(
                  phoneme: word.phonemes[i],
                  weakest: i == word.weakestPhonemeIndex &&
                      word.phonemes.length > 1,
                ),
            ],
          ),
          if (word.weakestPhoneme != null && word.phonemes.length > 1) ...[
            const SizedBox(height: 10),
            Row(
              children: [
                const Icon(Icons.flag_rounded,
                    color: SayColors.weak, size: 16),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    'Focus on the “${word.weakestPhoneme!.phoneme}” sound',
                    style: TextStyle(
                        color: Colors.white.withValues(alpha: 0.7),
                        fontSize: 12),
                  ),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }
}

class _PhonemePill extends StatelessWidget {
  const _PhonemePill({required this.phoneme, required this.weakest});
  final PhonemeScore phoneme;
  final bool weakest;

  @override
  Widget build(BuildContext context) {
    final color = SayColors.forScore(phoneme.score);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.16),
        borderRadius: BorderRadius.circular(10),
        border: weakest ? Border.all(color: SayColors.weak, width: 2) : null,
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(phoneme.phoneme,
              style: TextStyle(
                  color: color, fontSize: 14, fontWeight: FontWeight.w800)),
          const SizedBox(width: 5),
          Text('${phoneme.score.round()}',
              style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.6), fontSize: 11)),
        ],
      ),
    );
  }
}

class _WhatWeHeard extends StatelessWidget {
  const _WhatWeHeard({required this.phonemes});
  final List<String> phonemes;

  @override
  Widget build(BuildContext context) {
    return Theme(
      data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
      child: ExpansionTile(
        tilePadding: EdgeInsets.zero,
        childrenPadding: const EdgeInsets.only(bottom: 8),
        title: Text('What we heard',
            style: TextStyle(
                color: Colors.white.withValues(alpha: 0.6), fontSize: 13)),
        children: [
          Align(
            alignment: Alignment.centerLeft,
            child: Text(
              phonemes.isEmpty ? '(silence)' : phonemes.join(' '),
              style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.75),
                  fontSize: 13,
                  height: 1.5),
            ),
          ),
        ],
      ),
    );
  }
}
