import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';

import '../models/detection.dart';
import '../services/melange_service.dart';
import '../services/preprocessor.dart';
import '../theme.dart';
import '../widgets/detection_overlay.dart';
import '../widgets/hud_bar.dart';
import '../widgets/stats_bar.dart';

/// Bundled demo photos (demo_validation ORIGINAL images). Picking one runs the
/// exact same pipeline as a gallery photo — no worksite/live camera required.
const List<_Sample> _kSamples = [
  _Sample('assets/samples/original_img_009.jpg', 'Sample 1'),
  _Sample('assets/samples/original_img_023.jpg', 'Sample 2'),
  _Sample('assets/samples/original_img_027.jpg', 'Sample 3'),
];

class _Sample {
  const _Sample(this.asset, this.label);
  final String asset;
  final String label;
}

/// Main screen (static-image pivot): pick a photo from the gallery or run a
/// bundled sample, then show the PPE detection overlay + per-class counts +
/// inference latency on that still image.
class DetectScreen extends StatefulWidget {
  const DetectScreen({super.key, required this.service});

  final MelangeService service;

  @override
  State<DetectScreen> createState() => _DetectScreenState();
}

class _DetectScreenState extends State<DetectScreen> {
  final ImagePicker _picker = ImagePicker();

  bool _busy = false;
  bool _debugHud = false;
  String? _error;

  // The image currently displayed (upright pixels) + its dimensions.
  Uint8List? _displayBytes;
  int _imgW = 0;
  int _imgH = 0;

  List<Detection> _detections = const [];
  InferenceResult? _lastResult;

  @override
  void dispose() {
    widget.service.dispose();
    super.dispose();
  }

  Future<void> _pickFromGallery() async {
    try {
      final XFile? picked =
          await _picker.pickImage(source: ImageSource.gallery);
      if (picked == null) return;
      final bytes = await picked.readAsBytes();
      await _runOnBytes(bytes);
    } catch (e) {
      if (mounted) setState(() => _error = 'Could not open image: $e');
    }
  }

  Future<void> _pickSample(_Sample sample) async {
    try {
      final data = await rootBundle.load(sample.asset);
      await _runOnBytes(data.buffer.asUint8List());
    } catch (e) {
      if (mounted) setState(() => _error = 'Could not load sample: $e');
    }
  }

  /// Decodes the JPEG/PNG to upright RGB and runs the SAME detection pipeline
  /// the camera path used — only the input source changed.
  Future<void> _runOnBytes(Uint8List bytes) async {
    if (_busy) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final decoded = img.decodeImage(bytes);
      if (decoded == null) {
        throw const FormatException('Unsupported or corrupt image');
      }
      // Apply EXIF orientation so pixels are display-upright (matches how
      // Flutter renders the same bytes below).
      final baked = img.bakeOrientation(decoded);
      final Uint8List bgra = baked.getBytes(order: img.ChannelOrder.bgra);

      final frame = FrameData.fromBgraImage(
        width: baked.width,
        height: baked.height,
        bgra: bgra,
      );
      final result = await widget.service.detect(frame);
      if (!mounted) return;
      setState(() {
        _displayBytes = bytes;
        _imgW = baked.width;
        _imgH = baked.height;
        _detections = result.detections;
        _lastResult = result;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Inference failed: $e';
        _detections = const [];
      });
      debugPrint('Inference error: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  void _openSamples() {
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: SiteColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) {
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Bundled samples',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 4),
                const Text(
                  'demo_validation photos, bundled with the app.',
                  style: TextStyle(color: Colors.white54, fontSize: 12),
                ),
                const SizedBox(height: 12),
                for (final s in _kSamples)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(10),
                      child: Material(
                        color: Colors.white10,
                        child: ListTile(
                          leading: ClipRRect(
                            borderRadius: BorderRadius.circular(6),
                            child: Image.asset(
                              s.asset,
                              width: 56,
                              height: 56,
                              fit: BoxFit.cover,
                            ),
                          ),
                          title: Text(
                            s.label,
                            style: const TextStyle(color: Colors.white),
                          ),
                          trailing: const Icon(Icons.chevron_right,
                              color: Colors.white54),
                          onTap: () {
                            Navigator.of(context).pop();
                            _pickSample(s);
                          },
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    var hardhat = 0, vest = 0, noHardhat = 0, noVest = 0;
    for (final d in _detections) {
      switch (d.classId) {
        case kClassHardhat:
          hardhat++;
        case kClassVest:
          vest++;
        case kClassNoHardhat:
          noHardhat++;
        case kClassNoVest:
          noVest++;
      }
    }

    final result = _lastResult;

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          const ColoredBox(color: SiteColors.background),

          // The chosen still image with the detection overlay mapped onto it.
          Positioned.fill(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(8, 96, 8, 150),
              child: Center(
                child: _displayBytes == null
                    ? const _EmptyState()
                    : AspectRatio(
                        aspectRatio: _imgH == 0 ? 1 : _imgW / _imgH,
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            ClipRRect(
                              borderRadius: BorderRadius.circular(10),
                              child: Image.memory(
                                _displayBytes!,
                                fit: BoxFit.fill,
                                gaplessPlayback: true,
                              ),
                            ),
                            // sensorOrientation 0: still images are already
                            // upright, so the overlay maps normalized boxes
                            // directly onto the displayed image rect.
                            CustomPaint(
                              painter: DetectionOverlay(
                                detections: _detections,
                                imageWidth: _imgW,
                                imageHeight: _imgH,
                                sensorOrientation: 0,
                              ),
                            ),
                          ],
                        ),
                      ),
              ),
            ),
          ),

          // Top HUD (latency readout).
          Align(
            alignment: Alignment.topCenter,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                HudBar(
                  preMs: result?.preMs ?? 0,
                  runMs: result?.runMs ?? 0,
                  postMs: result?.postMs ?? 0,
                  onSettingsTap: _openSettings,
                  onDebugTap: () => setState(() => _debugHud = !_debugHud),
                  debugOn: _debugHud,
                ),
                if (_debugHud && result != null)
                  _DebugLine(result: result, detections: _detections),
                if (_error != null) _ErrorLine(message: _error!),
              ],
            ),
          ),

          // Bottom: per-class counts + source controls.
          Align(
            alignment: Alignment.bottomCenter,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                StatsBar(
                  hardhatCount: hardhat,
                  vestCount: vest,
                  noHardhatCount: noHardhat,
                  noVestCount: noVest,
                ),
                _SourceBar(
                  busy: _busy,
                  onGallery: _pickFromGallery,
                  onSamples: _openSamples,
                ),
              ],
            ),
          ),

          if (_busy)
            const Positioned.fill(
              child: ColoredBox(
                color: Color(0x66000000),
                child: Center(child: CircularProgressIndicator()),
              ),
            ),
        ],
      ),
    );
  }

  void _openSettings() {
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: SiteColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) {
        return Padding(
          padding: const EdgeInsets.fromLTRB(20, 16, 20, 28),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Colors.white24,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              const SizedBox(height: 18),
              const Text(
                'Detection thresholds',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 12),
              const Text(
                'Per-class thresholds are fixed to the Stage-0 validated '
                'values:\n'
                '  Hardhat 0.25 · Vest 0.15 · NO-Hardhat 0.15 · NO-Vest 0.15\n\n'
                'These were measured head-to-head on a labeled PPE test set '
                '(see model_selection.md); a live slider would silently change '
                'the validated operating point, so it is intentionally absent.',
                style: TextStyle(color: Colors.white70, fontSize: 13),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _SourceBar extends StatelessWidget {
  const _SourceBar({
    required this.busy,
    required this.onGallery,
    required this.onSamples,
  });

  final bool busy;
  final VoidCallback onGallery;
  final VoidCallback onSamples;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      top: false,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(12, 0, 12, 10),
        child: Row(
          children: [
            Expanded(
              child: FilledButton.icon(
                onPressed: busy ? null : onGallery,
                icon: const Icon(Icons.photo_library_outlined),
                label: const Text('Gallery'),
                style: FilledButton.styleFrom(
                  backgroundColor: SiteColors.accent,
                  foregroundColor: Colors.black,
                  padding: const EdgeInsets.symmetric(vertical: 14),
                ),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: busy ? null : onSamples,
                icon: const Icon(Icons.collections_outlined),
                label: const Text('Samples'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: Colors.white,
                  side: const BorderSide(color: Colors.white24),
                  padding: const EdgeInsets.symmetric(vertical: 14),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState();

  @override
  Widget build(BuildContext context) {
    return const Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(Icons.image_search, color: Colors.white38, size: 72),
        SizedBox(height: 16),
        Text(
          'Pick a photo or run a bundled sample\nto check PPE compliance',
          textAlign: TextAlign.center,
          style: TextStyle(color: Colors.white54, fontSize: 15),
        ),
      ],
    );
  }
}

class _ErrorLine extends StatelessWidget {
  const _ErrorLine({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.fromLTRB(12, 6, 12, 0),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: SiteColors.scrim,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.redAccent, width: 1),
      ),
      child: Text(
        message,
        style: const TextStyle(color: Colors.white70, fontSize: 12),
      ),
    );
  }
}

/// On-screen diagnostics — the only observability that works in a release
/// device build (Dart print does not reach the native console there).
class _DebugLine extends StatelessWidget {
  const _DebugLine({required this.result, required this.detections});

  final InferenceResult result;
  final List<Detection> detections;

  @override
  Widget build(BuildContext context) {
    final first = detections.isEmpty ? null : detections.first;
    final rawStr = first == null
        ? 'none'
        : '${first.label} ${first.confidence.toStringAsFixed(2)} '
            '(${first.rect.left.toStringAsFixed(2)},'
            '${first.rect.top.toStringAsFixed(2)},'
            '${first.rect.right.toStringAsFixed(2)},'
            '${first.rect.bottom.toStringAsFixed(2)})';
    return Container(
      margin: const EdgeInsets.fromLTRB(12, 6, 12, 0),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: SiteColors.scrim,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        'buf=${result.bufWidth}x${result.bufHeight} '
        'rot=${result.rotationDegrees} '
        'src=${result.srcWidth}x${result.srcHeight} '
        'n=${detections.length} det0=$rawStr',
        style: const TextStyle(
          color: Colors.white70,
          fontSize: 10,
          fontFamily: 'monospace',
        ),
      ),
    );
  }
}
