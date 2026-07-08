import 'package:flutter/material.dart';

import '../state/app_state.dart';
import '../theme.dart';

class LoadingView extends StatelessWidget {
  const LoadingView({super.key, required this.state});
  final AppState state;

  @override
  Widget build(BuildContext context) {
    final downloading = state.phase == Phase.downloadingModel;
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Palette.bg1, Palette.bg0],
          ),
        ),
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 64,
                height: 64,
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [Palette.teal, Palette.cyan],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  borderRadius: BorderRadius.circular(18),
                  boxShadow: [
                    BoxShadow(
                      color: Palette.teal.withValues(alpha: 0.25),
                      blurRadius: 24,
                      spreadRadius: 2,
                    ),
                  ],
                ),
                child: const Icon(
                  Icons.monitor_heart,
                  size: 34,
                  color: Color(0xFF06121A),
                ),
              ),
              const SizedBox(height: 24),
              Text(
                downloading ? 'Preparing on-device model' : 'Loading',
                style: const TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                  color: Palette.textHi,
                ),
              ),
              const SizedBox(height: 6),
              const Text(
                'EchoNet · DeepLabV3 · ZETIC Melange',
                style: TextStyle(fontSize: 11.5, color: Palette.textMid),
              ),
              const SizedBox(height: 22),
              SizedBox(
                width: 220,
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(3),
                  child: LinearProgressIndicator(
                    value: downloading && state.downloadProgress > 0
                        ? state.downloadProgress
                        : null,
                    minHeight: 5,
                    backgroundColor: Palette.stroke,
                    valueColor: const AlwaysStoppedAnimation(Palette.teal),
                  ),
                ),
              ),
              if (downloading) ...[
                const SizedBox(height: 10),
                Text(
                  '${(state.downloadProgress * 100).toStringAsFixed(0)}%',
                  style: numStyle(size: 12, color: Palette.textMid),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class ErrorView extends StatelessWidget {
  const ErrorView({super.key, required this.message});
  final String message;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(28),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.error_outline, size: 44, color: Palette.amber),
              const SizedBox(height: 16),
              Text(
                message,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Palette.textMid, fontSize: 13),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
