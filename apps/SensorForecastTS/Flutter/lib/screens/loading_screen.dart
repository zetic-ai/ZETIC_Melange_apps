import 'package:flutter/material.dart';

import '../services/melange_service.dart';
import 'main_screen.dart';

/// Downloads (first launch) and warms up the Melange model, then hands off to
/// the live demo. On conference Wi-Fi the download is the slow part — the
/// progress bar is honest about it.
class LoadingScreen extends StatefulWidget {
  const LoadingScreen({super.key});

  @override
  State<LoadingScreen> createState() => _LoadingScreenState();
}

class _LoadingScreenState extends State<LoadingScreen> {
  final MelangeService _melange = MelangeService();
  double _progress = 0;
  String _stage = 'Connecting to ZETIC Melange…';
  String? _error;

  @override
  void initState() {
    super.initState();
    _boot();
  }

  Future<void> _boot() async {
    try {
      await _melange.init(onProgress: (p) {
        if (!mounted) return;
        setState(() {
          _progress = p;
          _stage = 'Downloading model… ${(p * 100).toStringAsFixed(0)}%';
        });
      });
      if (!mounted) return;
      setState(() => _stage = 'Warming up on-device backend…');
      _melange.warmUp();
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => MainScreen(melange: _melange)),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    }
  }

  @override
  void dispose() {
    // MainScreen takes ownership on success; only clean up on failure paths.
    if (_error != null) _melange.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF070C16),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 40),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.multiline_chart,
                  size: 64, color: Color(0xFF3FE0C5)),
              const SizedBox(height: 16),
              const Text(
                'SentryWave',
                style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.w700,
                    color: Colors.white),
              ),
              const SizedBox(height: 6),
              const Text(
                'On-device sensor forecasting & anomaly detection',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 13, color: Color(0xFF8899AA)),
              ),
              const SizedBox(height: 32),
              if (_error == null) ...[
                LinearProgressIndicator(
                  value: _progress > 0 && _progress < 1 ? _progress : null,
                  color: const Color(0xFF3FE0C5),
                  backgroundColor: const Color(0xFF1D2A40),
                ),
                const SizedBox(height: 12),
                Text(_stage,
                    style: const TextStyle(
                        fontSize: 12, color: Color(0xFFB9C6D8))),
              ] else ...[
                const Icon(Icons.error_outline,
                    color: Color(0xFFFF5470), size: 32),
                const SizedBox(height: 8),
                Text(
                  'Model load failed:\n$_error',
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                      fontSize: 12, color: Color(0xFFFF5470)),
                ),
                const SizedBox(height: 16),
                OutlinedButton(
                  onPressed: () {
                    setState(() {
                      _error = null;
                      _progress = 0;
                      _stage = 'Retrying…';
                    });
                    _boot();
                  },
                  child: const Text('Retry'),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
