import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'screens/loading_screen.dart';

/// Melange model coordinates (registered on the dashboard).
const String kModelName = 'ajayshah/AerialDetectYOLO';
const int kModelVersion = 1;

/// Personal key is injected at build time, never committed:
///   `flutter build ios --release --dart-define=ZETIC_KEY=YOUR_KEY`
const String kZeticKey = String.fromEnvironment('ZETIC_KEY');

List<CameraDescription> gCameras = <CameraDescription>[];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    gCameras = await availableCameras();
  } catch (_) {
    gCameras = <CameraDescription>[];
  }
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
