import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../services/ctc_decoder.dart';
import '../services/melange_service.dart';
import '../services/pipeline_isolate.dart';
import '../theme.dart';
import 'main_screen.dart';

/// Shown while Melange downloads + initializes BOTH models (detector then
/// recognizer), the pipeline isolate spawns, and the CTC charset loads. Owns
/// those resources and hands them to the main screen once ready.
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
    setState(() {
      _error = null;
      _progress = 0;
      _status = 'Loading text detector…';
    });
    try {
      // Charset for the CTC decoder: [blank] + en_dict.txt + ' '.
      final dictRaw = await rootBundle.loadString('assets/en_dict.txt');
      final decoder = CtcDecoder.fromDictString(dictRaw);

      final pipeline = await DocPipeline.spawn();

      // Two download phases: detector 0–50%, recognizer 50–100%.
      await _service.init(
        onDetectorProgress: (p) => _setProgress(
            p.clamp(0.0, 1.0) * 0.5, 'Loading text detector…'),
        onRecognizerProgress: (p) => _setProgress(
            0.5 + p.clamp(0.0, 1.0) * 0.5, 'Loading text recognizer…'),
      );

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

  void _setProgress(double p, String status) {
    if (!mounted) return;
    setState(() {
      _progress = p;
      _status = '$status ${(p * 100).toStringAsFixed(0)}%';
    });
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
                  Icons.remove_red_eye_outlined,
                  color: RedactColors.accent,
                  size: 96,
                ),
              ),
              const SizedBox(height: 20),
              RichText(
                text: const TextSpan(
                  children: [
                    TextSpan(
                      text: 'Redact',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 34,
                        fontWeight: FontWeight.w900,
                        letterSpacing: 1,
                      ),
                    ),
                    TextSpan(
                      text: 'Lens',
                      style: TextStyle(
                        color: RedactColors.accent,
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
                'Live document PII redaction — fully on-device',
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
                  _status,
                  style: const TextStyle(color: Colors.white70, fontSize: 14),
                ),
              ] else
                _ErrorBlock(error: _error!, onRetry: _load),
              const SizedBox(height: 40),
              Text(
                'Powered by ZETIC Melange',
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
  const _ErrorBlock({required this.error, required this.onRetry});

  final String error;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        const Icon(Icons.error_outline, color: Colors.redAccent, size: 36),
        const SizedBox(height: 12),
        const Text(
          'Could not load the models',
          style: TextStyle(
            color: Colors.white,
            fontSize: 16,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          error,
          textAlign: TextAlign.center,
          maxLines: 6,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(color: Colors.white54, fontSize: 12),
        ),
        const SizedBox(height: 16),
        FilledButton.icon(
          onPressed: onRetry,
          icon: const Icon(Icons.refresh),
          label: const Text('Retry'),
          style: FilledButton.styleFrom(backgroundColor: RedactColors.accent),
        ),
      ],
    );
  }
}
