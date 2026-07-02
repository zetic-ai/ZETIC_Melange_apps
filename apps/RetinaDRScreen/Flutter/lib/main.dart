import 'package:flutter/material.dart';

import 'screens/loading_screen.dart';
import 'theme.dart';

void main() {
  runApp(const FundusGateApp());
}

class FundusGateApp extends StatelessWidget {
  const FundusGateApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FundusGate',
      debugShowCheckedModeBanner: false,
      theme: FundusTheme.build(),
      home: const LoadingScreen(),
    );
  }
}
