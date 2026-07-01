import 'package:flutter/material.dart';

import '../services/melange_service.dart';
import '../theme.dart';
import 'camera_screen.dart';
import 'photo_screen.dart';

/// Hosts the two detection modes and owns the shared [MelangeService] lifecycle
/// (created by the loading screen, closed here). A Live/Photo segmented toggle
/// swaps the body: only ONE mode is mounted at a time, so the live camera
/// stream and the one-shot photo detection never contend for the single busy
/// inference isolate.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, required this.service});

  final MelangeService service;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

enum _Mode { live, photo }

class _HomeScreenState extends State<HomeScreen> {
  _Mode _mode = _Mode.live;

  @override
  void dispose() {
    // Owner of the service tears it down (CameraScreen only disposes its own
    // camera controller now).
    widget.service.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Keyed so switching modes fully mounts/disposes each screen (camera
          // controller released when leaving Live; picker state reset likewise).
          _mode == _Mode.live
              ? CameraScreen(key: const ValueKey('live'), service: widget.service)
              : PhotoScreen(key: const ValueKey('photo'), service: widget.service),
          SafeArea(
            child: Align(
              alignment: Alignment.bottomCenter,
              child: Padding(
                padding: const EdgeInsets.only(bottom: 16),
                child: _ModeToggle(
                  mode: _mode,
                  onChanged: (m) => setState(() => _mode = m),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ModeToggle extends StatelessWidget {
  const _ModeToggle({required this.mode, required this.onChanged});

  final _Mode mode;
  final ValueChanged<_Mode> onChanged;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: AppTheme.surface.withValues(alpha: 0.9),
        borderRadius: BorderRadius.circular(28),
        border: Border.all(color: AppTheme.accentSoft),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          _segment('Live', Icons.videocam_outlined, _Mode.live),
          _segment('Photo', Icons.photo_outlined, _Mode.photo),
        ],
      ),
    );
  }

  Widget _segment(String label, IconData icon, _Mode value) {
    final selected = mode == value;
    return GestureDetector(
      onTap: () => onChanged(value),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
        decoration: BoxDecoration(
          color: selected ? AppTheme.accent : Colors.transparent,
          borderRadius: BorderRadius.circular(24),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              size: 18,
              color: selected ? AppTheme.bg : AppTheme.textMuted,
            ),
            const SizedBox(width: 8),
            Text(
              label,
              style: TextStyle(
                color: selected ? AppTheme.bg : AppTheme.textMuted,
                fontWeight: FontWeight.w700,
                fontSize: 14,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
