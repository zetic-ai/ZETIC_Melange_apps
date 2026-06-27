import 'package:flutter/material.dart';

import 'screens/loading_screen.dart';

/// ZETIC personal key. Injected at build time (GATE-2 decision 5):
///   `flutter build ios --release --dart-define=MLANGE_KEY=YOUR_KEY`
/// NEVER commit a real key. The default sentinel makes a forgotten define
/// fail loudly on the loading screen instead of silently.
const String kMlangeKey =
    String.fromEnvironment('MLANGE_KEY', defaultValue: 'YOUR_MLANGE_KEY');

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const VoxScribeApp());
}

class VoxScribeApp extends StatelessWidget {
  const VoxScribeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'VoxScribe',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF6C5CE7),
          brightness: Brightness.dark,
        ),
        scaffoldBackgroundColor: const Color(0xFF0B0F14),
      ),
      home: const LoadingScreen(personalKey: kMlangeKey),
    );
  }
}
