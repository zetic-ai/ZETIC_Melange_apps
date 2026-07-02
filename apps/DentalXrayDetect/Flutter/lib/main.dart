import 'package:flutter/material.dart';

import 'config/secrets.dart';
import 'screens/loading_screen.dart';
import 'theme.dart';

/// Registered Melange model name — **`ajayshah/DentalXRayDetect`**: the SDK
/// requires the fully-qualified `account/project` form (account `ajayshah`,
/// project `DentalXRayDetect` with a CAPITAL "R"). A bare project name fails
/// on-device with `MlangeException(3): Model name must include account name and
/// project name separated by slash(/)`. Note the project part DIFFERS from the
/// app folder name `DentalXrayDetect` (lowercase "r"). Pass this string EXACTLY.
const String kModelName = 'ajayshah/DentalXRayDetect';
const int kModelVersion = 1;

/// ZETIC personal key. Read from the gitignored lib/config/secrets.dart (see
/// secrets.example.dart); never committed.
const String kZeticKey = zeticPersonalKey;

/// User-facing product display name.
const String kProductName = 'OraLens';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const OraLensApp());
}

class OraLensApp extends StatelessWidget {
  const OraLensApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: kProductName,
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        colorScheme: const ColorScheme.dark(
          primary: AppTheme.accent,
          surface: AppTheme.surface,
        ),
        scaffoldBackgroundColor: AppTheme.bg,
        useMaterial3: true,
      ),
      home: const LoadingScreen(),
    );
  }
}
