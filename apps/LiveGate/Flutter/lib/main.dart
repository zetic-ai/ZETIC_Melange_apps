import 'package:flutter/material.dart';

import 'screens/loading_screen.dart';
import 'theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const TrueFaceApp());
}

class TrueFaceApp extends StatelessWidget {
  const TrueFaceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'TrueFace',
      debugShowCheckedModeBanner: false,
      theme: buildGateTheme(),
      home: const LoadingScreen(),
    );
  }
}
