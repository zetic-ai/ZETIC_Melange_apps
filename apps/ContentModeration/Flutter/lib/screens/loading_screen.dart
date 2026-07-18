import 'package:flutter/material.dart';

import '../services/melange_service.dart';
import '../theme.dart';
import 'main_screen.dart';

/// First screen: download (first launch) + warm up the Melange model behind a
/// progress bar, then hand off to the main moderation screen.
class LoadingScreen extends StatefulWidget {
  const LoadingScreen({super.key});

  @override
  State<LoadingScreen> createState() => _LoadingScreenState();
}

class _LoadingScreenState extends State<LoadingScreen> {
  final MelangeService _service = MelangeService();
  double _progress = 0;
  String _status = 'Preparing on-device model…';
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      await _service.load(
        onProgress: (p) {
          if (!mounted) return;
          setState(() {
            _progress = p;
            _status = p >= 1.0
                ? 'Warming up…'
                : 'Downloading model… ${(p * 100).toStringAsFixed(0)}%';
          });
        },
      );
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => MainScreen(service: _service)),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 40),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.shield_moon_outlined,
                  size: 72, color: SafeLensTheme.accent),
              const SizedBox(height: 20),
              const Text(
                'SafeLens',
                style: TextStyle(
                  color: SafeLensTheme.onSurface,
                  fontSize: 30,
                  fontWeight: FontWeight.w800,
                  letterSpacing: 0.5,
                ),
              ),
              const SizedBox(height: 6),
              const Text(
                'On-device content-safety gate',
                style: TextStyle(
                    color: SafeLensTheme.onSurfaceMuted, fontSize: 14),
              ),
              const SizedBox(height: 36),
              if (_error != null)
                Text(
                  'Could not load the model.\n$_error',
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                      color: Color(0xFFD2382C), fontSize: 13),
                )
              else ...[
                ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: LinearProgressIndicator(
                    value: _progress > 0 && _progress < 1 ? _progress : null,
                    minHeight: 6,
                    backgroundColor: SafeLensTheme.surfaceMuted,
                  ),
                ),
                const SizedBox(height: 14),
                Text(
                  _status,
                  style: const TextStyle(
                      color: SafeLensTheme.onSurfaceMuted, fontSize: 13),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
