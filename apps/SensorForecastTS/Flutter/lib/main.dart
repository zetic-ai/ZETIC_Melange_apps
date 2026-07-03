import 'package:flutter/material.dart';

import 'screens/loading_screen.dart';

void main() {
  runApp(const SentryWaveApp());
}

class SentryWaveApp extends StatelessWidget {
  const SentryWaveApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SentryWave',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF3FE0C5),
          brightness: Brightness.dark,
        ),
        scaffoldBackgroundColor: const Color(0xFF070C16),
        useMaterial3: true,
      ),
      home: const LoadingScreen(),
    );
  }
}
