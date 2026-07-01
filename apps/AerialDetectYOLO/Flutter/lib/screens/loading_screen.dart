import 'package:flutter/material.dart';

import '../main.dart';
import '../services/melange_service.dart';
import 'main_screen.dart';

/// Downloads + warms up the Melange model, then hands off to the upload screen.
class LoadingScreen extends StatefulWidget {
  const LoadingScreen({super.key});

  @override
  State<LoadingScreen> createState() => _LoadingScreenState();
}

class _LoadingScreenState extends State<LoadingScreen> {
  final MelangeService _service = MelangeService();
  String _status = 'Preparing…';
  String? _error;

  @override
  void initState() {
    super.initState();
    _start();
  }

  Future<void> _start() async {
    setState(() {
      _status = 'Downloading & optimizing model…';
      _error = null;
    });
    try {
      await _service.init(
        personalKey: kZeticKey,
        modelName: kModelName,
        version: kModelVersion,
      );
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute<void>(
          builder: (_) => MainScreen(service: _service),
        ),
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
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              const Icon(Icons.flight, size: 64, color: Color(0xFF34C759)),
              const SizedBox(height: 16),
              const Text(
                'SkyScout',
                style: TextStyle(fontSize: 26, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 4),
              const Text(
                'VisDrone YOLOv8s · 928 · on-device',
                style: TextStyle(color: Colors.white54, fontSize: 13),
              ),
              const SizedBox(height: 40),
              if (_error == null) ...<Widget>[
                ValueListenableBuilder<double>(
                  valueListenable: _service.progress,
                  builder: (_, double p, _) {
                    return Column(
                      children: <Widget>[
                        LinearProgressIndicator(
                          value: p > 0 && p < 1 ? p : null,
                          backgroundColor: Colors.white12,
                        ),
                        const SizedBox(height: 12),
                        Text(
                          p > 0 ? '${(p * 100).toStringAsFixed(0)}%' : _status,
                          style: const TextStyle(color: Colors.white70),
                        ),
                      ],
                    );
                  },
                ),
              ] else ...<Widget>[
                const Icon(Icons.error_outline, color: Colors.redAccent),
                const SizedBox(height: 12),
                Text(
                  _error!,
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.redAccent, fontSize: 12),
                ),
                const SizedBox(height: 16),
                FilledButton(onPressed: _start, child: const Text('Retry')),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
