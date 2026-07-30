import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;

import '../config.dart';
import '../services/ctc_decoder.dart';
import '../services/melange_service.dart';
import '../services/pipeline_isolate.dart';
import '../theme.dart';
import 'main_screen.dart';

/// Boot screen: downloads + warms BOTH Melange models (two downloads on a
/// fresh install — staged progress), loads the CTC charset asset, and spawns
/// the pipeline isolate. Hands everything to [MainScreen] once ready.
class LoadingScreen extends StatefulWidget {
  const LoadingScreen({super.key});

  @override
  State<LoadingScreen> createState() => _LoadingScreenState();
}

class _LoadingScreenState extends State<LoadingScreen>
    with SingleTickerProviderStateMixin {
  final MelangeService _service = MelangeService();
  late final AnimationController _pulse;

  double _progress = 0;
  String _status = 'Initializing…';
  String? _error;
  bool _keyMissing = false;

  @override
  void initState() {
    super.initState();
    _pulse = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1100),
    )..repeat(reverse: true);
    _load();
  }

  Future<void> _load() async {
    if (kMelangePersonalKey.isEmpty) {
      setState(() {
        _keyMissing = true;
        _error = 'This build has no Melange personal key.\n\n'
            'Rebuild with:\nflutter build ios --release '
            '--dart-define=MELANGE_PERSONAL_KEY=<your key>';
      });
      return;
    }
    setState(() {
      _error = null;
      _keyMissing = false;
      _progress = 0;
      _status = 'Loading models…';
    });
    try {
      // Charset first (fast, local): fails loudly here if the 838-class map
      // is broken rather than mid-demo.
      final charsetText =
          await rootBundle.loadString('assets/charset/latin_charset.txt');
      final decoder = CtcDecoder.fromCharsetText(charsetText);

      await _service.init(
        onStatus: (stage, p) {
          if (!mounted) return;
          setState(() {
            _progress = p.clamp(0.0, 1.0);
            _status = stage;
          });
        },
      );

      final pipeline = await PipelineWorker.spawn();

      if (!mounted) {
        pipeline.dispose();
        _service.dispose();
        return;
      }
      setState(() => _status = 'Ready');
      await Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (_) => MainScreen(
            service: _service,
            pipeline: pipeline,
            decoder: decoder,
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  @override
  void dispose() {
    _pulse.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 36),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ScaleTransition(
                scale: Tween(begin: 0.92, end: 1.08).animate(
                  CurvedAnimation(parent: _pulse, curve: Curves.easeInOut),
                ),
                child: const Icon(
                  Icons.translate_rounded,
                  color: GlyphColors.accent,
                  size: 92,
                ),
              ),
              const SizedBox(height: 20),
              RichText(
                text: const TextSpan(
                  children: [
                    TextSpan(
                      text: 'Glyph',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 34,
                        fontWeight: FontWeight.w900,
                        letterSpacing: 1,
                      ),
                    ),
                    TextSpan(
                      text: 'Go',
                      style: TextStyle(
                        color: GlyphColors.accent,
                        fontSize: 34,
                        fontWeight: FontWeight.w900,
                        letterSpacing: 1,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 6),
              Text(
                'Read any sign, anywhere — no signal needed',
                style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.6),
                  fontSize: 14,
                ),
              ),
              const SizedBox(height: 40),
              if (_error == null) ...[
                ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: LinearProgressIndicator(
                    value: _progress > 0 ? _progress : null,
                    minHeight: 8,
                    backgroundColor: Colors.white12,
                  ),
                ),
                const SizedBox(height: 14),
                Text(
                  '$_status ${(_progress * 100).toStringAsFixed(0)}%',
                  style: const TextStyle(color: Colors.white70, fontSize: 14),
                ),
              ] else
                _ErrorBlock(
                  error: _error!,
                  keyMissing: _keyMissing,
                  onRetry: _load,
                ),
              const SizedBox(height: 40),
              Text(
                'Powered by ZETIC Melange · 2 on-device models',
                style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.35),
                  fontSize: 12,
                  letterSpacing: 0.5,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ErrorBlock extends StatelessWidget {
  const _ErrorBlock({
    required this.error,
    required this.keyMissing,
    required this.onRetry,
  });

  final String error;
  final bool keyMissing;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Icon(
          keyMissing ? Icons.key_off_rounded : Icons.error_outline,
          color: keyMissing ? GlyphColors.warn : Colors.redAccent,
          size: 36,
        ),
        const SizedBox(height: 12),
        Text(
          keyMissing ? 'Melange key missing' : 'Could not load the models',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 16,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          error,
          textAlign: TextAlign.center,
          maxLines: 8,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(color: Colors.white54, fontSize: 12),
        ),
        if (!keyMissing) ...[
          const SizedBox(height: 16),
          FilledButton.icon(
            onPressed: onRetry,
            icon: const Icon(Icons.refresh),
            label: const Text('Retry'),
            style: FilledButton.styleFrom(backgroundColor: GlyphColors.accent),
          ),
        ],
      ],
    );
  }
}
