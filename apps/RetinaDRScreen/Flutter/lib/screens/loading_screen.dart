import 'package:flutter/material.dart';

import '../services/melange_service.dart';
import '../theme.dart';
import '../widgets/offline_badge.dart';
import 'main_screen.dart';

/// Downloads + warms up the Melange model, then hands off to [MainScreen].
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
    _init();
  }

  Future<void> _init() async {
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
        MaterialPageRoute(
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
          padding: const EdgeInsets.symmetric(horizontal: 32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.remove_red_eye_outlined,
                  size: 64, color: FundusTheme.primary),
              const SizedBox(height: 16),
              const Text(
                'FundusGate',
                style: TextStyle(
                  fontSize: 30,
                  fontWeight: FontWeight.w800,
                  letterSpacing: 0.5,
                ),
              ),
              const SizedBox(height: 4),
              const Text(
                'On-device diabetic-retinopathy screening',
                style: TextStyle(
                  color: FundusTheme.onSurfaceMuted,
                  fontSize: 13,
                ),
              ),
              const SizedBox(height: 40),
              if (_error == null) ...[
                ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: LinearProgressIndicator(
                    value: _progress > 0 && _progress < 1 ? _progress : null,
                    minHeight: 6,
                    backgroundColor: FundusTheme.surfaceAlt,
                    color: FundusTheme.primary,
                  ),
                ),
                const SizedBox(height: 14),
                Text(
                  _status,
                  style: const TextStyle(
                    color: FundusTheme.onSurfaceMuted,
                    fontSize: 13,
                  ),
                ),
              ] else
                _ErrorView(message: _error!),
              const SizedBox(height: 40),
              const OfflineBadge(),
            ],
          ),
        ),
      ),
    );
  }
}

class _ErrorView extends StatelessWidget {
  const _ErrorView({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        const Icon(Icons.cloud_off_rounded,
            color: FundusTheme.referable, size: 32),
        const SizedBox(height: 10),
        const Text(
          'Could not load the model',
          style: TextStyle(
            color: FundusTheme.referable,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          message,
          textAlign: TextAlign.center,
          style: const TextStyle(
            color: FundusTheme.onSurfaceMuted,
            fontSize: 12,
          ),
        ),
      ],
    );
  }
}
