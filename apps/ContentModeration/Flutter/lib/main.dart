import 'package:flutter/material.dart';

import 'screens/loading_screen.dart';
import 'theme.dart';

void main() {
  runApp(const SafeLensApp());
}

class SafeLensApp extends StatelessWidget {
  const SafeLensApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SafeLens',
      debugShowCheckedModeBanner: false,
      theme: SafeLensTheme.build(),
      home: const LoadingScreen(),
    );
  }
}
