import 'package:flutter/material.dart';

import '../main.dart';
import '../services/melange_service.dart';
import '../theme.dart';
import '../widgets/disclaimer_banner.dart';
import 'main_screen.dart';

/// Downloads + warms up the Melange model, then hands off to the analyzer.
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
      body: SafeArea(
        child: Column(
          children: <Widget>[
            Expanded(
              child: Center(
                child: Padding(
                  padding: const EdgeInsets.all(32),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: <Widget>[
                      const Icon(Icons.biotech,
                          size: 64, color: AppTheme.accent),
                      const SizedBox(height: 16),
                      const Text(
                        kProductName,
                        style: TextStyle(
                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                          color: AppTheme.textPrimary,
                        ),
                      ),
                      const SizedBox(height: 4),
                      const Text(
                        'Dental radiograph analyzer · YOLO11n · on-device',
                        style:
                            TextStyle(color: AppTheme.textMuted, fontSize: 13),
                      ),
                      const SizedBox(height: 40),
                      if (_error == null)
                        ValueListenableBuilder<double>(
                          valueListenable: _service.progress,
                          builder: (_, double p, _) {
                            return Column(
                              children: <Widget>[
                                LinearProgressIndicator(
                                  value: p > 0 && p < 1 ? p : null,
                                  backgroundColor: AppTheme.accentSoft,
                                  color: AppTheme.accent,
                                ),
                                const SizedBox(height: 12),
                                Text(
                                  p > 0
                                      ? '${(p * 100).toStringAsFixed(0)}%'
                                      : _status,
                                  style: const TextStyle(
                                      color: AppTheme.textMuted),
                                ),
                              ],
                            );
                          },
                        )
                      else ...<Widget>[
                        const Icon(Icons.error_outline, color: AppTheme.warn),
                        const SizedBox(height: 12),
                        Text(
                          _error!,
                          textAlign: TextAlign.center,
                          style: const TextStyle(
                              color: AppTheme.warn, fontSize: 12),
                        ),
                        const SizedBox(height: 16),
                        FilledButton(
                            onPressed: _start, child: const Text('Retry')),
                      ],
                    ],
                  ),
                ),
              ),
            ),
            const DisclaimerBanner(),
          ],
        ),
      ),
    );
  }
}
