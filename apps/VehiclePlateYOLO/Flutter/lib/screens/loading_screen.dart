import 'package:flutter/material.dart';

import '../services/melange_service.dart';
import '../theme.dart';
import 'camera_screen.dart';

/// Boots the Melange model (download + warm-up inside the dedicated isolate),
/// shows progress, then hands the live service to the camera screen.
class LoadingScreen extends StatefulWidget {
  const LoadingScreen({super.key});

  // Registered Melange model (GATE 0). Bump _modelVersion if ZETIC publishes a
  // new version; an in-place recompile re-downloads automatically.
  static const String modelName = 'ajayshah/VehiclePlateYOLO';
  static const int modelVersion = 1;

  // Injected at build time: --dart-define=ZETIC_KEY=<key>. Never hardcoded.
  static const String zeticKey = String.fromEnvironment('ZETIC_KEY');

  @override
  State<LoadingScreen> createState() => _LoadingScreenState();
}

class _LoadingScreenState extends State<LoadingScreen> {
  late final MelangeService _service;
  String _status = 'Preparing model…';
  String? _error;

  @override
  void initState() {
    super.initState();
    _service = MelangeService(
      modelName: LoadingScreen.modelName,
      modelVersion: LoadingScreen.modelVersion,
    );
    _boot();
  }

  Future<void> _boot() async {
    if (LoadingScreen.zeticKey.isEmpty) {
      setState(() {
        _error =
            'Missing ZETIC key. Build with --dart-define=ZETIC_KEY=<your_key>.';
      });
      return;
    }
    try {
      setState(() => _status = 'Downloading & warming up model…');
      await _service.init(personalKey: LoadingScreen.zeticKey);
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute<void>(
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
    // If we never navigated away (error path), tear the isolate down.
    if (_error != null) _service.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.bg,
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(
                Icons.directions_car_filled,
                size: 72,
                color: AppTheme.accent,
              ),
              const SizedBox(height: 24),
              const Text(
                'PlateHawk',
                style: TextStyle(
                  fontSize: 26,
                  fontWeight: FontWeight.w700,
                  color: AppTheme.textPrimary,
                  letterSpacing: 0.5,
                ),
              ),
              const SizedBox(height: 8),
              const Text(
                'On-device license-plate detection',
                style: TextStyle(color: AppTheme.textMuted),
              ),
              const SizedBox(height: 40),
              if (_error == null) ...[
                const SizedBox(
                  width: 28,
                  height: 28,
                  child: CircularProgressIndicator(
                    strokeWidth: 3,
                    color: AppTheme.accent,
                  ),
                ),
                const SizedBox(height: 20),
                Text(
                  _status,
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: AppTheme.textMuted),
                ),
              ] else
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: AppTheme.surface,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: AppTheme.warn.withValues(alpha: 0.5)),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.error_outline, color: AppTheme.warn),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          _error!,
                          style: const TextStyle(color: AppTheme.textPrimary),
                        ),
                      ),
                    ],
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
