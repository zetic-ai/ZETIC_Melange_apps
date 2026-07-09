import 'package:flutter/material.dart';

import 'screens/loading_screen.dart';
import 'theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const SayRightApp());
}

class SayRightApp extends StatelessWidget {
  const SayRightApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SayRight',
      debugShowCheckedModeBanner: false,
      theme: buildSayRightTheme(),
      home: const LoadingScreen(),
    );
  }
}
