import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../services/shelf_scanner.dart';
import '../theme.dart';
import '../widgets/detection_overlay.dart';
import '../widgets/hud.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({super.key, required this.scanner});

  final ShelfScanner scanner;

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  final ImagePicker _picker = ImagePicker();

  Uint8List? _imageBytes;
  ScanResult? _result;
  bool _busy = false;
  String? _error;
  bool _showDebug = true;

  Future<void> _pickFromGallery() async {
    try {
      final file = await _picker.pickImage(source: ImageSource.gallery);
      if (file == null) return;
      final bytes = await file.readAsBytes();
      await _runScan(bytes);
    } catch (e) {
      _setError('$e');
    }
  }

  Future<void> _runScan(Uint8List bytes) async {
    setState(() {
      _busy = true;
      _error = null;
      _imageBytes = bytes;
      _result = null;
    });
    // Yield one frame so the "Detecting…" state paints before the sync hot path.
    await Future<void>.delayed(Duration.zero);
    try {
      final result = widget.scanner.scan(bytes);
      if (!mounted) return;
      setState(() {
        _result = result;
        _busy = false;
      });
    } catch (e) {
      _setError('$e');
    }
  }

  void _setError(String message) {
    if (!mounted) return;
    setState(() {
      _error = message;
      _busy = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ShelfSense',
            style: TextStyle(fontWeight: FontWeight.w800)),
        actions: [
          IconButton(
            tooltip: 'Toggle debug HUD',
            icon: Icon(_showDebug ? Icons.bug_report : Icons.bug_report_outlined),
            onPressed: () => setState(() => _showDebug = !_showDebug),
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(child: _buildViewport()),
          if (_result != null && _showDebug) DebugHud(result: _result!),
          _buildControls(),
        ],
      ),
    );
  }

  Widget _buildViewport() {
    return Container(
      color: Colors.black,
      width: double.infinity,
      child: Stack(
        children: [
          if (_imageBytes != null && _result != null)
            Positioned.fill(
              child: DetectionOverlay(
                imageBytes: _imageBytes!,
                imageWidth: _result!.imageWidth,
                imageHeight: _result!.imageHeight,
                detections: _result!.detections,
              ),
            )
          else if (_imageBytes != null)
            Positioned.fill(
              child: Image.memory(_imageBytes!, fit: BoxFit.contain),
            )
          else
            _buildEmptyState(),
          if (_result != null && !_busy)
            Positioned(
              left: 16,
              top: 16,
              child: DetectionHud(result: _result!),
            ),
          if (_busy)
            Container(
              color: Colors.black.withValues(alpha: 0.55),
              alignment: Alignment.center,
              child: const Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  CircularProgressIndicator(color: ShelfSenseTheme.accent),
                  SizedBox(height: 14),
                  Text('Detecting products…'),
                ],
              ),
            ),
          if (_error != null)
            Positioned(
              left: 16,
              right: 16,
              bottom: 16,
              child: Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.red.withValues(alpha: 0.85),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(_error!, style: const TextStyle(fontSize: 12)),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.shelves,
              size: 64, color: Colors.white.withValues(alpha: 0.3)),
          const SizedBox(height: 12),
          Text(
            'Upload a shelf photo to detect products',
            style: TextStyle(color: Colors.white.withValues(alpha: 0.6)),
          ),
        ],
      ),
    );
  }

  Widget _buildControls() {
    return SafeArea(
      top: false,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(12, 10, 12, 8),
        child: SizedBox(
          width: double.infinity,
          child: FilledButton.icon(
            onPressed: _busy ? null : _pickFromGallery,
            icon: const Icon(Icons.photo_library),
            label: const Text('Upload shelf photo'),
          ),
        ),
      ),
    );
  }
}
