import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../models/moderation_result.dart';
import '../services/melange_service.dart';
import '../services/preprocessor.dart';
import '../theme.dart';
import '../widgets/diagnostics_hud.dart';
import '../widgets/offline_badge.dart';
import '../widgets/score_meter.dart';
import '../widgets/verdict_banner.dart';

/// The live demo: pick an image from the gallery OR camera, run one on-device
/// inference, and show the KEEP / REVIEW-BLUR / BLOCK decision. The picked image
/// is blurred in the preview for REVIEW and BLOCK — that blur-before-upload is
/// the demo story.
class MainScreen extends StatefulWidget {
  const MainScreen({super.key, required this.service});

  final MelangeService service;

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  final ImagePicker _picker = ImagePicker();

  Uint8List? _imageBytes;
  ModerationResult? _result;
  double _preprocessMs = 0;
  double _inferenceMs = 0;
  int _decodedW = 0;
  int _decodedH = 0;
  bool _busy = false;
  String? _error;

  Future<void> _pick(ImageSource source) async {
    if (_busy) return;
    try {
      final XFile? file = await _picker.pickImage(source: source);
      if (file == null) return;
      final bytes = await file.readAsBytes();
      await _moderate(bytes);
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    }
  }

  /// Preprocess (off the UI isolate) then run one inference on the UI isolate
  /// (the SDK binds the model handle to its creating isolate).
  Future<void> _moderate(Uint8List bytes) async {
    setState(() {
      _busy = true;
      _error = null;
      _imageBytes = bytes;
      _result = null;
    });
    try {
      final watch = Stopwatch()..start();
      final PreprocessOutput pre = await compute(preprocessWithDims, bytes);
      watch.stop();
      final outcome = widget.service.infer(pre.data);
      if (!mounted) return;
      setState(() {
        _preprocessMs = watch.elapsedMicroseconds / 1000.0;
        _inferenceMs = outcome.inferenceMs;
        _decodedW = pre.width;
        _decodedH = pre.height;
        _result = outcome.result;
        _busy = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = '$e';
        _busy = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final result = _result;
    return Scaffold(
      appBar: AppBar(title: const Text('SafeLens')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 28),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Center(child: OnDeviceBadge()),
              const SizedBox(height: 16),
              _ImagePanel(bytes: _imageBytes, busy: _busy, result: result),
              const SizedBox(height: 16),
              Row(
                children: [
                  Expanded(
                    child: FilledButton.icon(
                      onPressed: _busy ? null : () => _pick(ImageSource.gallery),
                      icon: const Icon(Icons.photo_library_outlined),
                      label: const Text('Gallery'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: FilledButton.icon(
                      onPressed: _busy ? null : () => _pick(ImageSource.camera),
                      icon: const Icon(Icons.photo_camera_outlined),
                      label: const Text('Camera'),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              if (_error != null)
                _ErrorBanner(message: _error!)
              else if (result != null) ...[
                VerdictBanner(result: result),
                const SizedBox(height: 16),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: ScoreMeter(result: result),
                  ),
                ),
                const SizedBox(height: 16),
                DiagnosticsHud(
                  result: result,
                  preprocessMs: _preprocessMs,
                  inferenceMs: _inferenceMs,
                  decodedWidth: _decodedW,
                  decodedHeight: _decodedH,
                ),
              ] else
                const _EmptyHint(),
            ],
          ),
        ),
      ),
    );
  }
}

class _ImagePanel extends StatelessWidget {
  const _ImagePanel({required this.bytes, required this.busy, this.result});

  final Uint8List? bytes;
  final bool busy;
  final ModerationResult? result;

  @override
  Widget build(BuildContext context) {
    final shouldBlur = result?.decision.shouldBlur ?? false;
    return AspectRatio(
      aspectRatio: 1,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: Container(
          color: SafeLensTheme.surface,
          child: Stack(
            fit: StackFit.expand,
            children: [
              if (bytes != null)
                Image.memory(bytes!, fit: BoxFit.contain)
              else
                const Center(
                  child: Icon(Icons.add_photo_alternate_outlined,
                      size: 56, color: SafeLensTheme.onSurfaceMuted),
                ),
              // Blur-before-upload overlay for REVIEW / BLOCK.
              if (bytes != null && shouldBlur)
                ClipRect(
                  child: BackdropFilter(
                    filter: ui.ImageFilter.blur(sigmaX: 18, sigmaY: 18),
                    child: Container(
                      color: (result?.decision.color ?? Colors.black)
                          .withValues(alpha: 0.18),
                      alignment: Alignment.center,
                      child: Icon(
                        result?.decision == Decision.block
                            ? Icons.block
                            : Icons.visibility_off_outlined,
                        color: Colors.white,
                        size: 44,
                      ),
                    ),
                  ),
                ),
              if (busy)
                Container(
                  color: Colors.black54,
                  child: const Center(child: CircularProgressIndicator()),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _EmptyHint extends StatelessWidget {
  const _EmptyHint();

  @override
  Widget build(BuildContext context) {
    return const Padding(
      padding: EdgeInsets.symmetric(vertical: 12),
      child: Text(
        'Pick an image from your gallery or camera to screen it for unsafe '
        'content on-device, before it ever leaves the phone.',
        textAlign: TextAlign.center,
        style: TextStyle(color: SafeLensTheme.onSurfaceMuted, fontSize: 13),
      ),
    );
  }
}

class _ErrorBanner extends StatelessWidget {
  const _ErrorBanner({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFD2382C).withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFD2382C)),
      ),
      child: Text(
        message,
        style: const TextStyle(color: Color(0xFFD2382C), fontSize: 13),
      ),
    );
  }
}
