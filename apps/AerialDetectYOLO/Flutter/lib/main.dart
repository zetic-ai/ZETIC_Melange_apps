import 'dart:async';

import 'package:flutter/material.dart';

import 'screens/loading_screen.dart';
import 'services/sample_seed.dart';

/// Melange model coordinates (registered on the dashboard).
const String kModelName = 'ajayshah/AerialDetectYOLO';
const int kModelVersion = 1;

/// Personal key is injected at build time, never committed:
///   `flutter build ios --release --dart-define=ZETIC_KEY=YOUR_KEY`
const String kZeticKey = String.fromEnvironment('ZETIC_KEY');

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Silently seed the bundled demo image into the device photo library once
  // (never blocks or crashes app start — errors are swallowed internally).
  unawaited(saveSampleToPhotosOnce());
  runApp(const AerialDetectApp());
}

class AerialDetectApp extends StatelessWidget {
  const AerialDetectApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SkyScout',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF34C759),
          surface: Color(0xFF101418),
        ),
        scaffoldBackgroundColor: const Color(0xFF0B0E11),
        useMaterial3: true,
      ),
      home: const LoadingScreen(),
    );
  }
}
