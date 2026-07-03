import 'dart:async';
import 'dart:collection';

import 'package:flutter/material.dart';

import '../models/forecast.dart';
import '../services/data_feed.dart';
import '../services/melange_service.dart';
import '../services/pipeline.dart';
import '../services/postprocessor.dart';
import '../widgets/event_list.dart';
import '../widgets/hud.dart';
import '../widgets/live_chart.dart';

/// Samples per second the demo clock replays (GATE-2 approved default).
const double kSamplesPerSecond = 20.0;

/// Visible history samples on the chart.
const int kVisibleHistory = 280;

class MainScreen extends StatefulWidget {
  const MainScreen({super.key, required this.melange});

  final MelangeService melange;

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  /// Owned by the screen so the slider's threshold survives feed resets.
  final AnomalyDetector _detector = AnomalyDetector();

  late SensorFeed _feed;
  late ForecastPipeline _pipeline;
  Timer? _clock;
  bool _running = true;
  bool _busy = false; // one inference in flight max; drop, never queue
  FeedMode _mode = FeedMode.industrial;

  final Queue<double> _visible = Queue<double>();
  int _visibleStart = 0;
  final List<AnomalyEvent> _events = [];
  final List<int> _anomalyMarks = [];

  int _revision = 0;
  double _runMs = 0;
  double _pipelineUs = 0;
  double? _lastScore;

  @override
  void initState() {
    super.initState();
    _resetFeed(FeedMode.industrial);
    _clock = Timer.periodic(
      Duration(microseconds: (1e6 / kSamplesPerSecond).round()),
      (_) => _tick(),
    );
  }

  /// Seeds a fresh feed + pipeline and PRE-FILLS the full 512-sample window
  /// (full-window contract: the model never sees a partial or padded window,
  /// and the demo starts forecasting on the first visible tick).
  void _resetFeed(FeedMode mode) {
    _mode = mode;
    _feed = SensorFeed(mode: mode);
    _detector.reset();
    _pipeline = ForecastPipeline(detector: _detector);
    _visible.clear();
    _events.clear();
    _anomalyMarks.clear();
    _lastScore = null;
    for (var i = 0; i < 512; i++) {
      final v = _feed.next();
      _pipeline.push(v);
      _pushVisible(v);
    }
    _visibleStart = _pipeline.globalIndex - _visible.length;
    _revision++;
  }

  void _pushVisible(double v) {
    _visible.addLast(v);
    while (_visible.length > kVisibleHistory) {
      _visible.removeFirst();
      _visibleStart++;
    }
  }

  void _tick() {
    if (!_running || !mounted) return;

    final sw = Stopwatch()..start();
    final v = _feed.next();
    final res = _pipeline.push(v);
    _pushVisible(v);
    _visibleStart = _pipeline.globalIndex - _visible.length;
    _lastScore = res.score;

    if (res.flagged) {
      _anomalyMarks.add(res.globalIndex);
      if (_anomalyMarks.length > 200) _anomalyMarks.removeAt(0);
      // Record one event per debounce-fire onset (streak == debounceCount).
      if (_pipeline.detector.streak == _pipeline.detector.debounceCount) {
        _events.add(AnomalyEvent(
          globalIndex: res.globalIndex,
          wallClock: DateTime.now(),
          value: v,
          score: res.score ?? 0,
        ));
      }
    }
    sw.stop();
    _pipelineUs = sw.elapsedMicroseconds.toDouble();

    if (res.needForecast && !_busy) {
      _busy = true;
      try {
        final out = widget.melange.run(_pipeline.window);
        _pipeline.applyForecast(out.raw);
        _runMs = out.runMs;
      } finally {
        _busy = false;
      }
    }

    setState(() => _revision++);
  }

  @override
  void dispose() {
    _clock?.cancel();
    widget.melange.dispose(); // model.close() — this screen owns the model
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final forecast = _pipeline.forecast;
    final data = ChartData(
      revision: _revision,
      history: _visible.toList(growable: false),
      historyStartIndex: _visibleStart,
      forecast: forecast,
      anomalyIndices: _anomalyMarks,
      threshold: _pipeline.detector.threshold,
    );

    return Scaffold(
      backgroundColor: const Color(0xFF070C16),
      appBar: AppBar(
        backgroundColor: const Color(0xFF0B1220),
        title: const Text('SentryWave',
            style: TextStyle(fontWeight: FontWeight.w600)),
        actions: [
          IconButton(
            tooltip: _running ? 'Pause' : 'Resume',
            icon: Icon(_running ? Icons.pause : Icons.play_arrow),
            onPressed: () => setState(() => _running = !_running),
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            // Mode switch.
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 8, 12, 0),
              child: SegmentedButton<FeedMode>(
                segments: const [
                  ButtonSegment(
                      value: FeedMode.industrial,
                      icon: Icon(Icons.factory_outlined, size: 16),
                      label: Text('Machine replay')),
                  ButtonSegment(
                      value: FeedMode.lab,
                      icon: Icon(Icons.science_outlined, size: 16),
                      label: Text('Lab signal')),
                ],
                selected: {_mode},
                onSelectionChanged: (s) => setState(() => _resetFeed(s.first)),
              ),
            ),
            // Chart + HUD overlay.
            Expanded(
              flex: 5,
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Stack(
                    children: [
                      Positioned.fill(child: LiveChart(data: data)),
                      Positioned(
                        left: 8,
                        top: 8,
                        child: Hud(
                          runMs: _runMs,
                          pipelineUs: _pipelineUs,
                          score: _lastScore,
                          threshold: _pipeline.detector.threshold,
                          eventCount: _events.length,
                          samplesPerSecond: kSamplesPerSecond,
                          modelLine:
                              '${MelangeService.modelName} v${MelangeService.modelVersion} '
                              'RUN_AUTO (served artifact: native console)',
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            // Injection buttons + threshold slider.
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              child: Row(
                children: [
                  _injectButton('Spike', Icons.bolt, InjectionKind.spike),
                  const SizedBox(width: 8),
                  _injectButton(
                      'Shift', Icons.stairs_outlined, InjectionKind.levelShift),
                  const SizedBox(width: 8),
                  _injectButton('Noise', Icons.waves_outlined,
                      InjectionKind.noiseBurst),
                  const Spacer(),
                  const Text('thr',
                      style:
                          TextStyle(color: Color(0xFF8899AA), fontSize: 12)),
                  Slider(
                    value: _pipeline.detector.threshold,
                    min: 0.5,
                    max: 3.0,
                    divisions: 25,
                    label:
                        _pipeline.detector.threshold.toStringAsFixed(1),
                    onChanged: (v) =>
                        setState(() => _pipeline.detector.threshold = v),
                  ),
                ],
              ),
            ),
            // Event list.
            Expanded(
              flex: 2,
              child: Container(
                margin: const EdgeInsets.fromLTRB(12, 0, 12, 12),
                decoration: BoxDecoration(
                  color: const Color(0xFF0B1220),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: EventList(events: _events),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _injectButton(String label, IconData icon, InjectionKind kind) {
    return OutlinedButton.icon(
      style: OutlinedButton.styleFrom(
        foregroundColor: const Color(0xFFFF9E64),
        side: const BorderSide(color: Color(0xFF3A2A20)),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      ),
      onPressed: () => _feed.inject(kind),
      icon: Icon(icon, size: 16),
      label: Text(label, style: const TextStyle(fontSize: 12)),
    );
  }
}
