import 'dart:async';

import 'package:flutter/material.dart';

import 'screens/loading_screen.dart';
import 'services/demo_seed.dart';
import 'theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Fire-and-forget: silently place the bundled demo image into the photo
  // library on first launch only (guarded + swallows all errors), so the user
  // can upload it. No UI, no button, no snackbar.
  unawaited(ensureDemoImageSaved());
  runApp(const VehiclePlateApp());
}

class VehiclePlateApp extends StatelessWidget {
  const VehiclePlateApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PlateHawk',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.dark,
      home: const LoadingScreen(),
    );
  }
}
