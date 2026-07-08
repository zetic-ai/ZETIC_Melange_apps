import 'package:flutter/material.dart';

import '../overlay/mask_overlay.dart';
import '../state/app_state.dart';
import '../theme.dart';
import '../video/clip_source.dart';
import 'benchmark_hud.dart';

class BenchmarkView extends StatelessWidget {
  const BenchmarkView({super.key, required this.state});
  final AppState state;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Palette.bg1, Palette.bg0],
          ),
        ),
        child: SafeArea(
          child: ListView(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 20),
            children: [
              _header(),
              const SizedBox(height: 14),
              _clipSelector(),
              const SizedBox(height: 12),
              // Video + HUD rebuild per-frame via frameTick; the controls above/below
              // do NOT (keeps their GestureDetectors stable so taps always register).
              ValueListenableBuilder<int>(
                valueListenable: state.frameTick,
                builder: (_, _, _) => _videoCard(),
              ),
              const SizedBox(height: 14),
              _pacingToggle(),
              const SizedBox(height: 14),
              ValueListenableBuilder<int>(
                valueListenable: state.frameTick,
                builder: (_, _, _) => BenchmarkHud(
                  snapshot: state.snapshot,
                  device: state.device,
                ),
              ),
              const SizedBox(height: 14),
              _attribution(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _header() {
    return Row(
      children: [
        Container(
          width: 40,
          height: 40,
          decoration: BoxDecoration(
            gradient: const LinearGradient(
              colors: [Palette.teal, Palette.cyan],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            borderRadius: BorderRadius.circular(11),
          ),
          child: const Icon(
            Icons.monitor_heart,
            size: 22,
            color: Color(0xFF06121A),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Cardiac Ultrasound Segmentation',
                style: TextStyle(
                  fontSize: 16.5,
                  fontWeight: FontWeight.w700,
                  color: Palette.textHi,
                ),
              ),
              Text(
                'On-device LV segmentation · capability proof',
                style: TextStyle(color: Palette.textMid, fontSize: 11.5),
              ),
            ],
          ),
        ),
        const Pill(color: Palette.green, dot: true, child: Text('ON-DEVICE')),
      ],
    );
  }

  Widget _clipSelector() {
    final clips = state.clips;
    if (clips.length < 2) return const SizedBox.shrink();
    return SizedBox(
      height: 62,
      child: Row(
        children: [
          for (var i = 0; i < clips.length; i++) ...[
            if (i > 0) const SizedBox(width: 10),
            Expanded(
              child: _clipChip(
                i,
                clips[i].thumbPath,
                clips[i].view,
                clips[i].label,
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _clipChip(int index, String thumb, String view, String label) {
    final selected = state.activeClip == index;
    return GestureDetector(
      onTap: () => state.selectClip(index),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 160),
        decoration: BoxDecoration(
          color: selected ? Palette.teal.withValues(alpha: 0.12) : Palette.card,
          borderRadius: BorderRadius.circular(11),
          border: Border.all(
            color: selected ? Palette.teal : Palette.stroke,
            width: selected ? 1.4 : 1,
          ),
        ),
        padding: const EdgeInsets.all(6),
        child: Row(
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(7),
              child: SizedBox(
                width: 46,
                height: 46,
                child: thumb.isEmpty
                    ? const ColoredBox(color: Palette.bg0)
                    : Image.asset(thumb, fit: BoxFit.cover),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    view,
                    style: TextStyle(
                      fontSize: 12.5,
                      fontWeight: FontWeight.w700,
                      color: selected ? Palette.teal : Palette.textHi,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 1),
                  Text(
                    label,
                    style: const TextStyle(
                      fontSize: 9.5,
                      color: Palette.textMid,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _videoCard() {
    return Panel(
      padding: const EdgeInsets.all(8),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(11),
        child: AspectRatio(
          aspectRatio: 1,
          child: Stack(
            fit: StackFit.expand,
            children: [
              if (state.currentFramePath != null)
                Image.asset(
                  state.currentFramePath!,
                  gaplessPlayback: true,
                  fit: BoxFit.cover,
                ),
              Positioned.fill(
                child: CustomPaint(painter: MaskPainter(state.maskImage)),
              ),
              // subtle vignette for depth
              const DecoratedBox(
                decoration: BoxDecoration(
                  gradient: RadialGradient(
                    radius: 0.9,
                    colors: [Colors.transparent, Color(0x330A0E14)],
                    stops: [0.7, 1.0],
                  ),
                ),
              ),
              _readoutChip(),
              _legendChip(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _readoutChip() {
    return Positioned(
      left: 10,
      top: 10,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(10),
        child: Container(
          width: 128,
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
          color: Colors.black.withValues(alpha: 0.42),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text('LV CAVITY', style: labelStyle(color: Palette.textMid)),
              const SizedBox(height: 2),
              Text(
                '${state.cavityAreaPx} px',
                style: numStyle(size: 18, color: Palette.textHi),
              ),
              const SizedBox(height: 4),
              Row(
                children: [
                  Text('FAC ', style: labelStyle(color: Palette.coral)),
                  SizedBox(
                    width: 34,
                    child: Text(
                      '${state.facPercent.toStringAsFixed(0)}%',
                      style: numStyle(size: 13, color: Palette.coral),
                    ),
                  ),
                  Text('EF proxy', style: labelStyle(color: Palette.textLow)),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _legendChip() {
    return Positioned(
      right: 10,
      bottom: 10,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(20),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          color: Colors.black.withValues(alpha: 0.42),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 8,
                height: 8,
                decoration: const BoxDecoration(
                  color: Palette.coral,
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 6),
              const Text(
                'LV segmentation',
                style: TextStyle(fontSize: 11, color: Palette.textHi),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _pacingToggle() {
    Widget seg(PacingMode mode, String label, IconData icon) {
      final selected = state.pacing == mode;
      return Expanded(
        child: GestureDetector(
          onTap: () => state.setPacing(mode),
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 160),
            padding: const EdgeInsets.symmetric(vertical: 10),
            decoration: BoxDecoration(
              color: selected
                  ? Palette.teal.withValues(alpha: 0.16)
                  : Colors.transparent,
              borderRadius: BorderRadius.circular(9),
              border: Border.all(
                color: selected
                    ? Palette.teal.withValues(alpha: 0.5)
                    : Colors.transparent,
              ),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  icon,
                  size: 14,
                  color: selected ? Palette.teal : Palette.textMid,
                ),
                const SizedBox(width: 7),
                Text(
                  label,
                  style: TextStyle(
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600,
                    color: selected ? Palette.teal : Palette.textMid,
                  ),
                ),
              ],
            ),
          ),
        ),
      );
    }

    return Panel(
      padding: const EdgeInsets.all(5),
      child: Row(
        children: [
          seg(PacingMode.benchmark, 'Benchmark', Icons.speed),
          const SizedBox(width: 5),
          seg(PacingMode.realtime, 'Realtime 24fps', Icons.play_circle_outline),
        ],
      ),
    );
  }

  Widget _attribution() {
    return Text(
      state.attribution,
      style: const TextStyle(color: Palette.textLow, fontSize: 9, height: 1.3),
      textAlign: TextAlign.center,
      maxLines: 2,
      overflow: TextOverflow.ellipsis,
    );
  }
}
