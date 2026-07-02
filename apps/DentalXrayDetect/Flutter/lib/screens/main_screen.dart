import 'dart:async';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show Uint8List;
import 'package:image_picker/image_picker.dart';

import '../main.dart';
import '../models/detection.dart';
import '../models/label.dart';
import '../services/image_decoder.dart';
import '../services/melange_service.dart';
import '../theme.dart';
import '../widgets/detection_overlay.dart';
import '../widgets/disclaimer_banner.dart';

/// The analyzer: upload a dental X-ray from the photo library, run the Melange
/// still pipeline ([MelangeService.inferStill]), and show the radiograph large
/// with class-colored boxes, a per-class count row, and a latency +
/// tensor-shape diagnostics footer. Upload is the ONLY input path — there are
/// no bundled sample radiographs. The non-diagnostic disclaimer is pinned to
/// the bottom in every state.
class MainScreen extends StatefulWidget {
  const MainScreen({super.key, required this.service});

  final MelangeService service;

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  final ImagePicker _picker = ImagePicker();

  ui.Image? _image;
  int _imageW = 0;
  int _imageH = 0;
  List<Detection> _detections = const <Detection>[];
  FrameTimings? _timings;

  bool _working = false;
  String? _error;

  final TransformationController _zoomController = TransformationController();
  double _viewScale = 1.0;

  @override
  void initState() {
    super.initState();
    _zoomController.addListener(_onZoomChanged);
  }

  void _onZoomChanged() {
    final double scale = _zoomController.value.getMaxScaleOnAxis();
    if (scale != _viewScale) {
      setState(() => _viewScale = scale);
    }
  }

  /// Open the photo library and analyze the chosen X-ray.
  Future<void> _pick() async {
    if (_working) return;
    setState(() {
      _working = true;
      _error = null;
    });
    try {
      final XFile? file = await _picker.pickImage(source: ImageSource.gallery);
      if (file == null) {
        if (mounted) setState(() => _working = false);
        return;
      }
      await _runOnBytes(await file.readAsBytes());
    } catch (e) {
      _fail(e);
    }
  }

  /// Decode (EXIF baked, off the UI isolate), build the display image, and run
  /// the same still detection path for every source.
  Future<void> _runOnBytes(Uint8List fileBytes) async {
    final DecodedImage? decoded = await decodeStillImage(fileBytes);
    if (decoded == null) {
      _fail('Could not decode that image.');
      return;
    }

    final ui.Image image = await _toUiImage(decoded);
    final result = await widget.service.inferStill(
      decoded.rgb,
      decoded.width,
      decoded.height,
    );
    if (!mounted) return;

    _image?.dispose();
    setState(() {
      _image = image;
      _imageW = decoded.width;
      _imageH = decoded.height;
      _detections = result?.detections ?? const <Detection>[];
      _timings = result?.timings;
      _working = false;
      _error = result == null ? 'Model not ready — try again.' : null;
    });
  }

  Future<ui.Image> _toUiImage(DecodedImage d) {
    final Completer<ui.Image> completer = Completer<ui.Image>();
    ui.decodeImageFromPixels(
      d.rgba,
      d.width,
      d.height,
      ui.PixelFormat.rgba8888,
      completer.complete,
    );
    return completer.future;
  }

  void _fail(Object e) {
    if (!mounted) return;
    setState(() {
      _working = false;
      _error = '$e';
    });
  }

  @override
  void dispose() {
    _zoomController.removeListener(_onZoomChanged);
    _zoomController.dispose();
    _image?.dispose();
    widget.service.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final ui.Image? image = _image;
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: <Widget>[
            _TopBar(hasImage: image != null, detections: _detections),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 8),
                child: image == null
                    ? _EmptyState(busy: _working, error: _error)
                    : InteractiveViewer(
                        transformationController: _zoomController,
                        minScale: 1.0,
                        maxScale: 8.0,
                        panEnabled: true,
                        scaleEnabled: true,
                        clipBehavior: Clip.hardEdge,
                        boundaryMargin: const EdgeInsets.all(120),
                        child: DetectionOverlay(
                          image: image,
                          detections: _detections,
                          viewScale: _viewScale,
                        ),
                      ),
              ),
            ),
            if (_error != null && image != null)
              Padding(
                padding: const EdgeInsets.fromLTRB(12, 6, 12, 0),
                child: Text(_error!,
                    style: const TextStyle(color: AppTheme.warn, fontSize: 12)),
              ),
            _Actions(working: _working, hasImage: image != null, onUpload: _pick),
            if (image != null)
              _DiagnosticsFooter(
                timings: _timings,
                imageW: _imageW,
                imageH: _imageH,
                detections: _detections,
              ),
            const DisclaimerBanner(),
          ],
        ),
      ),
    );
  }
}

/// Top bar: product name + total count, then a per-class count chip row.
class _TopBar extends StatelessWidget {
  const _TopBar({required this.hasImage, required this.detections});

  final bool hasImage;
  final List<Detection> detections;

  @override
  Widget build(BuildContext context) {
    // Per-class counts across all 3 classes, in canonical label order.
    final Map<int, int> counts = <int, int>{};
    for (final Detection d in detections) {
      counts[d.classId] = (counts[d.classId] ?? 0) + 1;
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
      decoration: const BoxDecoration(color: AppTheme.surface),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Row(
            children: <Widget>[
              const Icon(Icons.biotech, size: 20, color: AppTheme.accent),
              const SizedBox(width: 8),
              const Text(
                kProductName,
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: AppTheme.textPrimary,
                ),
              ),
              const Spacer(),
              if (hasImage)
                Text(
                  '${detections.length} finding'
                  '${detections.length == 1 ? '' : 's'}',
                  style: const TextStyle(
                      color: AppTheme.textMuted, fontSize: 13),
                ),
            ],
          ),
          if (hasImage) ...<Widget>[
            const SizedBox(height: 10),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: <Widget>[
                for (int c = 0; c < kNumClasses; c++)
                  _CountChip(classId: c, count: counts[c] ?? 0),
              ],
            ),
          ],
        ],
      ),
    );
  }
}

class _CountChip extends StatelessWidget {
  const _CountChip({required this.classId, required this.count});

  final int classId;
  final int count;

  @override
  Widget build(BuildContext context) {
    final Color color = colorForClass(classId);
    final bool on = count > 0;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: on ? 0.18 : 0.06),
        borderRadius: BorderRadius.circular(20),
        border:
            Border.all(color: color.withValues(alpha: on ? 0.9 : 0.3), width: 1.5),
      ),
      child: Text(
        '${prettyLabel(classId)} $count',
        style: TextStyle(
          color: on ? AppTheme.textPrimary : AppTheme.textMuted,
          fontSize: 13,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }
}

/// Bottom action row: primary "Upload X-ray" button.
class _Actions extends StatelessWidget {
  const _Actions({
    required this.working,
    required this.hasImage,
    required this.onUpload,
  });

  final bool working;
  final bool hasImage;
  final VoidCallback onUpload;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 8),
      child: SizedBox(
        width: double.infinity,
        child: FilledButton.icon(
          onPressed: working ? null : onUpload,
          style: FilledButton.styleFrom(
            backgroundColor: AppTheme.accent,
            foregroundColor: AppTheme.bg,
            padding: const EdgeInsets.symmetric(vertical: 14),
          ),
          icon: working
              ? const SizedBox(
                  width: 16,
                  height: 16,
                  child: CircularProgressIndicator(
                      strokeWidth: 2, color: AppTheme.bg),
                )
              : const Icon(Icons.upload_outlined),
          label: Text(
            working
                ? 'Analyzing…'
                : (hasImage ? 'Upload another X-ray' : 'Upload X-ray'),
            style: const TextStyle(fontWeight: FontWeight.w700),
          ),
        ),
      ),
    );
  }
}

/// On-screen diagnostics HUD. Because release-build Dart `print` does NOT reach
/// the native console, per-stage timings, the output tensor shape, and the raw
/// first box are surfaced here on the UI, not via logs.
class _DiagnosticsFooter extends StatelessWidget {
  const _DiagnosticsFooter({
    required this.timings,
    required this.imageW,
    required this.imageH,
    required this.detections,
  });

  final FrameTimings? timings;
  final int imageW;
  final int imageH;
  final List<Detection> detections;

  @override
  Widget build(BuildContext context) {
    final FrameTimings? t = timings;
    final String latency = t == null
        ? '—'
        : '${t.totalMs.toStringAsFixed(0)} ms '
            '(pre ${t.preprocessMs.toStringAsFixed(0)} · run '
            '${t.runMs.toStringAsFixed(0)} · post '
            '${t.postprocessMs.toStringAsFixed(0)})';
    final String firstBox = detections.isEmpty
        ? 'none'
        : '${detections.first.label} '
            '${(detections.first.confidence * 100).toStringAsFixed(0)}% '
            '[${detections.first.left.toStringAsFixed(0)},'
            '${detections.first.top.toStringAsFixed(0)},'
            '${detections.first.right.toStringAsFixed(0)},'
            '${detections.first.bottom.toStringAsFixed(0)}]';

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            'on-device · $latency',
            style: const TextStyle(
              color: AppTheme.textMuted,
              fontSize: 11,
              fontFeatures: <FontFeature>[FontFeature.tabularFigures()],
            ),
          ),
          Text(
            'in ${imageW}x$imageH → out [1,7,8400] · box1 $firstBox',
            style: const TextStyle(color: AppTheme.textMuted, fontSize: 10),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState({required this.busy, this.error});

  final bool busy;
  final String? error;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            Icon(
              error == null ? Icons.image_search_outlined : Icons.error_outline,
              size: 64,
              color: error == null
                  ? AppTheme.accent.withValues(alpha: busy ? 0.4 : 0.8)
                  : AppTheme.warn,
            ),
            const SizedBox(height: 14),
            Text(
              error ??
                  (busy
                      ? 'Analyzing…'
                      : 'Upload a dental X-ray from your photo library to '
                          'detect caries, periapical lesions & impacted '
                          'teeth.'),
              textAlign: TextAlign.center,
              style: TextStyle(
                color: error == null ? AppTheme.textMuted : AppTheme.warn,
                fontSize: 13,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
