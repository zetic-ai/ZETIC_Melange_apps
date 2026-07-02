import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../models/screening_result.dart';
import '../services/melange_service.dart';
import '../services/preprocessor.dart';
import '../theme.dart';
import '../widgets/confidence_bar.dart';
import '../widgets/diagnostics_hud.dart';
import '../widgets/disclaimer_card.dart';
import '../widgets/offline_badge.dart';
import '../widgets/verdict_banner.dart';

/// The live demo: UPLOAD a fundus image from the photo library, run one
/// on-device inference, and show the REFERABLE / NOT-REFERABLE verdict.
/// Upload-only — there is no bundled-sample flow.
class MainScreen extends StatefulWidget {
  const MainScreen({super.key, required this.service});

  final MelangeService service;

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  final ImagePicker _picker = ImagePicker();

  Uint8List? _imageBytes;
  ScreeningResult? _result;
  double _preprocessMs = 0;
  double _inferenceMs = 0;
  bool _busy = false;
  String? _error;

  Future<void> _pickFromGallery() async {
    try {
      final XFile? file =
          await _picker.pickImage(source: ImageSource.gallery);
      if (file == null) return;
      final bytes = await file.readAsBytes();
      await _screen(bytes);
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    }
  }

  /// Preprocess (off the UI isolate) then run one inference on the UI isolate
  /// (the SDK binds the model handle to its creating isolate).
  Future<void> _screen(Uint8List bytes) async {
    setState(() {
      _busy = true;
      _error = null;
      _imageBytes = bytes;
      _result = null;
    });
    try {
      final watch = Stopwatch()..start();
      final Float32List input = await compute(preprocessFundusBytes, bytes);
      watch.stop();
      final outcome = widget.service.infer(input);
      if (!mounted) return;
      setState(() {
        _preprocessMs = watch.elapsedMicroseconds / 1000.0;
        _inferenceMs = outcome.inferenceMs;
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
      appBar: AppBar(title: const Text('FundusGate')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Center(child: OfflineBadge()),
              const SizedBox(height: 16),
              _ImagePanel(bytes: _imageBytes, busy: _busy),
              const SizedBox(height: 16),
              FilledButton.icon(
                onPressed: _busy ? null : _pickFromGallery,
                icon: const Icon(Icons.photo_library_outlined),
                label: const Text('Upload image'),
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
                    child: ConfidenceBar(pReferable: result.pReferable),
                  ),
                ),
                const SizedBox(height: 16),
                DiagnosticsHud(
                  result: result,
                  preprocessMs: _preprocessMs,
                  inferenceMs: _inferenceMs,
                ),
                const SizedBox(height: 16),
              ] else
                const _EmptyHint(),
              const SizedBox(height: 4),
              const DisclaimerCard(),
            ],
          ),
        ),
      ),
    );
  }
}

class _ImagePanel extends StatelessWidget {
  const _ImagePanel({required this.bytes, required this.busy});

  final Uint8List? bytes;
  final bool busy;

  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: 1,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: Container(
          color: FundusTheme.surface,
          child: Stack(
            fit: StackFit.expand,
            children: [
              if (bytes != null)
                Image.memory(bytes!, fit: BoxFit.contain)
              else
                const Center(
                  child: Icon(Icons.image_outlined,
                      size: 56, color: FundusTheme.onSurfaceMuted),
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
        'Upload a color fundus image from your photo library to screen it '
        'on-device.',
        textAlign: TextAlign.center,
        style: TextStyle(color: FundusTheme.onSurfaceMuted, fontSize: 13),
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
        color: FundusTheme.referable.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: FundusTheme.referable),
      ),
      child: Text(
        message,
        style: const TextStyle(color: FundusTheme.referable, fontSize: 13),
      ),
    );
  }
}
