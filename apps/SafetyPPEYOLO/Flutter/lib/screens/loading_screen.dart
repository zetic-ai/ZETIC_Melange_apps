import 'package:flutter/material.dart';

import '../services/melange_service.dart';
import '../theme.dart';
import 'camera_screen.dart';

/// Shown while Melange downloads + initializes + warms up the model. Owns the
/// [MelangeService] and hands it to the camera screen once ready.
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
      _status = 'Loading model…';
    });
    try {
      await _service.init(
        onProgress: (p) {
          if (!mounted) return;
          setState(() {
            _progress = p.clamp(0.0, 1.0);
            _status = 'Loading model… ${(_progress * 100).toStringAsFixed(0)}%';
          });
        },
      );
      if (!mounted) return;
      setState(() => _status = 'Warmed up — ready');
      await Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (_) => CameraScreen(service: _service),
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
                  Icons.engineering,
                  color: SiteColors.accent,
                  size: 96,
                ),
              ),
              const SizedBox(height: 20),
              RichText(
                text: const TextSpan(
                  children: [
                    TextSpan(
                      text: 'Site',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 34,
                        fontWeight: FontWeight.w900,
                        letterSpacing: 1,
                      ),
                    ),
                    TextSpan(
                      text: 'Guard',
                      style: TextStyle(
                        color: SiteColors.accent,
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
                'Real-time PPE compliance detection',
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
          'Could not load the model',
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
          maxLines: 4,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(color: Colors.white54, fontSize: 12),
        ),
        const SizedBox(height: 16),
        FilledButton.icon(
          onPressed: onRetry,
          icon: const Icon(Icons.refresh),
          label: const Text('Retry'),
          style: FilledButton.styleFrom(backgroundColor: SiteColors.accent),
        ),
      ],
    );
  }
}
