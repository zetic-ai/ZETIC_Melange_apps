import 'package:flutter/material.dart';

import 'screens/loading_screen.dart';
import 'theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const PyroGuardApp());
}

class PyroGuardApp extends StatelessWidget {
  const PyroGuardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PyroGuard',
      debugShowCheckedModeBanner: false,
      theme: buildPyroGuardTheme(),
      home: const LoadingScreen(),
    );
  }
}
