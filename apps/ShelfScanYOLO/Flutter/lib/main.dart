import 'package:flutter/material.dart';

import 'screens/loading_screen.dart';
import 'theme.dart';

void main() {
  runApp(const ShelfSenseApp());
}

class ShelfSenseApp extends StatelessWidget {
  const ShelfSenseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ShelfSense',
      debugShowCheckedModeBanner: false,
      theme: ShelfSenseTheme.dark,
      home: const LoadingScreen(),
    );
  }
}
