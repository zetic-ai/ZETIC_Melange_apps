import 'dart:async';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show ByteData, Uint8List, rootBundle;
import 'package:gal/gal.dart';
import 'package:image_picker/image_picker.dart';

import '../models/detection.dart';
import '../models/label.dart';
import '../services/image_decoder.dart';
import '../services/melange_service.dart';
import '../widgets/photo_overlay.dart';

/// Bundled demo aerial image (a drone car-park shot) shipped with the app so the
/// detector can be exercised without a photo library — see pubspec `assets`. It
/// can also be saved into the device gallery via "Save demo photo to Photos".
const String _kSampleAsset = 'assets/samples/aerial_sample.jpg';

/// The demo, upload-only: the user picks an aerial/drone photo, it runs through
/// the Melange still pipeline ([MelangeService.inferStill]), and the result is
/// shown large with thin class-colored boxes and a compact per-class count bar.
///
/// There is no live camera mode: SkyScout is trained on top-down aerial imagery,
/// so a real drone photo is the only place it fires — a ground-level live feed
/// stays empty. The screen therefore opens straight onto the upload UI.
class MainScreen extends StatefulWidget {
  const MainScreen({super.key, required this.service});

  final MelangeService service;

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  final ImagePicker _picker = ImagePicker();

  ui.Image? _image;
  List<Detection> _detections = const <Detection>[];
  FrameTimings? _timings;

  bool _working = false;
  String? _error;

  /// Open the photo library and run detection on the chosen still.
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
      if (mounted) {
        setState(() {
          _working = false;
          _error = '$e';
        });
      }
    }
  }

  /// Decode (EXIF baked, off the UI isolate), build the display image, and run
  /// [MelangeService.inferStill] — the same still path for every picked photo.
  Future<void> _runOnBytes(Uint8List fileBytes) async {
    final DecodedImage? decoded = await decodeStillImage(fileBytes);
    if (decoded == null) {
      if (mounted) {
        setState(() {
          _working = false;
          _error = 'Could not decode that image.';
        });
      }
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

  /// Save the bundled demo photo into the device gallery so the user can then
  /// upload it (this replaces the removed "Try sample" in-app path).
  Future<void> _saveDemoToPhotos() async {
    if (_working) return;
    try {
      final ByteData data = await rootBundle.load(_kSampleAsset);
      final Uint8List bytes = data.buffer.asUint8List();
      if (!await Gal.hasAccess(toAlbum: true)) {
        await Gal.requestAccess(toAlbum: true);
      }
      await Gal.putImageBytes(bytes, name: 'skyscout_demo');
      _snack('Demo photo saved to Photos — now tap “Upload photo”.');
    } on GalException catch (e) {
      _snack('Could not save demo photo: ${e.type.message}');
    } catch (e) {
      _snack('Could not save demo photo: $e');
    }
  }

  void _snack(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..clearSnackBars()
      ..showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  void dispose() {
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
                    ? _EmptyState(error: _error)
                    : PhotoOverlay(image: image, detections: _detections),
              ),
            ),
            if (_timings != null && image != null) _LatencyFooter(_timings!),
            _Actions(
              working: _working,
              hasImage: image != null,
              onUpload: _pick,
              onSaveDemo: _saveDemoToPhotos,
            ),
          ],
        ),
      ),
    );
  }
}

/// Clean top bar: the app name plus, once a result exists, a compact per-class
/// count chip row ("car 137 · van 3 · truck 4"). Nothing overlaps the image.
class _TopBar extends StatelessWidget {
  const _TopBar({required this.hasImage, required this.detections});

  final bool hasImage;
  final List<Detection> detections;

  @override
  Widget build(BuildContext context) {
    // Per-class counts, most-common first.
    final Map<int, int> counts = <int, int>{};
    for (final Detection d in detections) {
      counts[d.classId] = (counts[d.classId] ?? 0) + 1;
    }
    final List<MapEntry<int, int>> ordered = counts.entries.toList()
      ..sort((MapEntry<int, int> a, MapEntry<int, int> b) =>
          b.value.compareTo(a.value));

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
      decoration: const BoxDecoration(color: Color(0xFF101418)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Row(
            children: <Widget>[
              const Icon(Icons.flight, size: 20, color: Color(0xFF34C759)),
              const SizedBox(width: 8),
              const Text(
                'SkyScout',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const Spacer(),
              if (hasImage)
                Text(
                  '${detections.length} object'
                  '${detections.length == 1 ? '' : 's'}',
                  style: const TextStyle(color: Colors.white54, fontSize: 13),
                ),
            ],
          ),
          if (ordered.isNotEmpty) ...<Widget>[
            const SizedBox(height: 10),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: <Widget>[
                for (final MapEntry<int, int> e in ordered)
                  _CountChip(classId: e.key, count: e.value),
              ],
            ),
          ] else if (hasImage) ...<Widget>[
            const SizedBox(height: 10),
            const Text(
              'No objects found — try a top-down aerial/drone photo.',
              style: TextStyle(color: Colors.white54, fontSize: 13),
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
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.18),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withValues(alpha: 0.9), width: 1.5),
      ),
      child: Text(
        '${labelForClass(classId)} $count',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 13,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }
}

/// A single tiny, unobtrusive latency line under the image.
class _LatencyFooter extends StatelessWidget {
  const _LatencyFooter(this.timings);

  final FrameTimings timings;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 4, bottom: 2),
      child: Text(
        '${timings.totalMs.toStringAsFixed(0)} ms on-device',
        style: const TextStyle(
          color: Colors.white38,
          fontSize: 11,
          fontFeatures: <FontFeature>[FontFeature.tabularFigures()],
        ),
      ),
    );
  }
}

/// Bottom action row: primary "Upload photo" (only real action) plus a
/// secondary "Save demo photo to Photos".
class _Actions extends StatelessWidget {
  const _Actions({
    required this.working,
    required this.hasImage,
    required this.onUpload,
    required this.onSaveDemo,
  });

  final bool working;
  final bool hasImage;
  final VoidCallback onUpload;
  final VoidCallback onSaveDemo;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              onPressed: working ? null : onUpload,
              icon: working
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.photo_library),
              label: Text(hasImage ? 'Pick another' : 'Upload photo'),
            ),
          ),
          const SizedBox(height: 8),
          TextButton.icon(
            onPressed: working ? null : onSaveDemo,
            icon: const Icon(Icons.download),
            label: const Text('Save demo photo to Photos'),
          ),
        ],
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState({this.error});

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
              error == null ? Icons.image_search : Icons.error_outline,
              size: 56,
              color: error == null ? Colors.white38 : Colors.redAccent,
            ),
            const SizedBox(height: 12),
            Text(
              error ??
                  'Upload an aerial or drone photo to run detection.\n'
                      'No photo handy? Save the demo photo to your gallery, '
                      'then upload it.',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: error == null ? Colors.white54 : Colors.redAccent,
                fontSize: 13,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
