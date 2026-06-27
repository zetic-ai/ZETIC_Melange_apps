import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;

import '../services/pipeline_isolate.dart';
import 'main_screen.dart';

/// Loads the bundled assets, then spawns the worker isolate that downloads and
/// warms all THREE models (segmentation + Whisper encoder + decoder), showing
/// per-stage cold-start progress. Hands the live controller + demo clip to
/// [MainScreen] once ready.
class LoadingScreen extends StatefulWidget {
  const LoadingScreen({super.key, required this.personalKey});

  final String personalKey;

  @override
  State<LoadingScreen> createState() => _LoadingScreenState();
}

class _LoadingScreenState extends State<LoadingScreen> {
  static const List<String> _stageNames = <String>[
    'Speaker segmentation',
    'Whisper encoder',
    'Whisper decoder',
  ];

  PipelineController? _controller;
  int _stage = 0;
  double _progress = 0;
  String? _error;

  @override
  void initState() {
    super.initState();
    _boot();
  }

  Future<void> _boot() async {
    if (widget.personalKey.isEmpty || widget.personalKey == 'YOUR_MLANGE_KEY') {
      setState(() => _error =
          'MLANGE_KEY is not set. Build with --dart-define=MLANGE_KEY=<your_key>.');
      return;
    }

    try {
      final ByteData wav = await rootBundle.load('assets/demo_2spk.wav');
      final ByteData mel = await rootBundle.load('assets/mel_filters_80.bin');
      final String vocab = await rootBundle.loadString('assets/vocab.json');

      final Uint8List wavBytes = wav.buffer.asUint8List(
        wav.offsetInBytes,
        wav.lengthInBytes,
      );
      // Copy out into an aligned Float32List for the isolate.
      final Float32List melFilters =
          Float32List(mel.lengthInBytes ~/ 4);
      for (int i = 0; i < melFilters.length; i++) {
        melFilters[i] = mel.getFloat32(i * 4, Endian.little);
      }

      final PipelineController controller = PipelineController(
        personalKey: widget.personalKey,
        melFilters: melFilters,
        vocabJson: vocab,
      );
      _controller = controller;
      controller.loadProgress.listen((({int stage, double progress}) e) {
        if (mounted) {
          setState(() {
            _stage = e.stage;
            _progress = e.progress;
          });
        }
      });

      await controller.start();

      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute<void>(
          builder: (_) => MainScreen(controller: controller, wavBytes: wavBytes),
        ),
      );
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    }
  }

  @override
  void dispose() {
    if (ModalRoute.of(context)?.isCurrent ?? false) {
      _controller?.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final ThemeData theme = Theme.of(context);
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              Icon(Icons.graphic_eq, size: 64, color: theme.colorScheme.primary),
              const SizedBox(height: 16),
              Text('VoxScribe', style: theme.textTheme.headlineSmall),
              const SizedBox(height: 8),
              Text(
                'On-device speaker-labeled transcript',
                style:
                    theme.textTheme.bodyMedium?.copyWith(color: Colors.white60),
              ),
              const SizedBox(height: 32),
              if (_error != null)
                Text(
                  _error!,
                  textAlign: TextAlign.center,
                  style: TextStyle(color: theme.colorScheme.error),
                )
              else ...<Widget>[
                LinearProgressIndicator(value: _progress > 0 ? _progress : null),
                const SizedBox(height: 12),
                Text(
                  'Loading ${_stageNames[_stage]} '
                  '(${_stage + 1}/3)… ${(_progress * 100).clamp(0, 100).toStringAsFixed(0)}%',
                  textAlign: TextAlign.center,
                  style:
                      theme.textTheme.bodySmall?.copyWith(color: Colors.white54),
                ),
                const SizedBox(height: 4),
                Text(
                  'First launch downloads 3 models over the network.',
                  textAlign: TextAlign.center,
                  style:
                      theme.textTheme.bodySmall?.copyWith(color: Colors.white30),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
