import 'package:flutter/material.dart';

import '../services/melange_service.dart';
import '../services/shelf_scanner.dart';
import '../theme.dart';
import 'main_screen.dart';

/// Downloads + warms up the Melange model, then hands a ready [ShelfScanner]
/// to the main screen. Shows download progress and a clear error+retry path
/// (the model download can fail on poor conference Wi-Fi or a bad key).
class LoadingScreen extends StatefulWidget {
  const LoadingScreen({super.key});

  @override
  State<LoadingScreen> createState() => _LoadingScreenState();
}

class _LoadingScreenState extends State<LoadingScreen> {
  final MelangeService _melange = MelangeService();
  double _progress = 0.0;
  String _status = 'Preparing ShelfSense…';
  String? _error;

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    setState(() {
      _error = null;
      _status = 'Downloading detection model…';
      _progress = 0.0;
    });
    try {
      await _melange.initialize(
        onProgress: (p) {
          if (mounted) setState(() => _progress = p.clamp(0.0, 1.0));
        },
      );
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (_) => MainScreen(scanner: ShelfScanner(melange: _melange)),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = '$e';
        _status = 'Could not load the model.';
      });
    }
  }

  @override
  void dispose() {
    // If we never navigated away, release the handle.
    if (_error != null) _melange.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.shelves, size: 72, color: ShelfSenseTheme.accent),
              const SizedBox(height: 16),
              const Text(
                'ShelfSense',
                style: TextStyle(fontSize: 30, fontWeight: FontWeight.w800),
              ),
              const SizedBox(height: 6),
              Text(
                'On-device shelf SKU detector',
                style: TextStyle(color: Colors.white.withValues(alpha: 0.6)),
              ),
              const SizedBox(height: 40),
              if (_error == null) ...[
                SizedBox(
                  width: 220,
                  child: LinearProgressIndicator(
                    value: _progress > 0 ? _progress : null,
                    minHeight: 6,
                    backgroundColor: ShelfSenseTheme.surfaceHigh,
                    color: ShelfSenseTheme.accent,
                  ),
                ),
                const SizedBox(height: 14),
                Text(_status,
                    style: TextStyle(color: Colors.white.withValues(alpha: 0.7))),
              ] else ...[
                Text(_status,
                    style: const TextStyle(
                        fontWeight: FontWeight.w600, fontSize: 16)),
                const SizedBox(height: 10),
                Text(
                  _error!,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    color: Colors.redAccent.withValues(alpha: 0.9),
                    fontSize: 12,
                  ),
                ),
                const SizedBox(height: 20),
                FilledButton(onPressed: _initialize, child: const Text('Retry')),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
