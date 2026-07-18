import 'package:flutter/material.dart';

import 'screens/loading_screen.dart';
import 'theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const SnapSeekApp());
}

class SnapSeekApp extends StatelessWidget {
  const SnapSeekApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SnapSeek',
      debugShowCheckedModeBanner: false,
      theme: buildSnapTheme(),
      home: const LoadingScreen(),
    );
  }
}
