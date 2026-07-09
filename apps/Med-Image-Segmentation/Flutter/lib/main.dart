import 'package:flutter/material.dart';

import 'state/app_state.dart';
import 'theme.dart';
import 'views/benchmark_view.dart';
import 'views/loading_view.dart';

void main() => runApp(const MedSegApp());

class MedSegApp extends StatefulWidget {
  const MedSegApp({super.key});

  @override
  State<MedSegApp> createState() => _MedSegAppState();
}

class _MedSegAppState extends State<MedSegApp> {
  final _state = AppState();

  @override
  void initState() {
    super.initState();
    _state.init();
  }

  @override
  void dispose() {
    _state.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Med Segmentation Benchmark',
      debugShowCheckedModeBanner: false,
      theme: buildTheme(),
      home: ListenableBuilder(
        listenable: _state,
        builder: (context, _) {
          switch (_state.phase) {
            case Phase.loadingAssets:
            case Phase.downloadingModel:
              return LoadingView(state: _state);
            case Phase.error:
              return ErrorView(message: _state.errorMessage);
            case Phase.running:
              return BenchmarkView(state: _state);
          }
        },
      ),
    );
  }
}
