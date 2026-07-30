import 'package:flutter/material.dart';

import 'screens/loading_screen.dart';
import 'theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const GlyphGoApp());
}

class GlyphGoApp extends StatelessWidget {
  const GlyphGoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'GlyphGo',
      debugShowCheckedModeBanner: false,
      theme: buildGlyphGoTheme(),
      home: const LoadingScreen(),
    );
  }
}
