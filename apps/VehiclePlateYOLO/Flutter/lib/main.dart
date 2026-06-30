import 'package:flutter/material.dart';

import 'screens/loading_screen.dart';
import 'theme.dart';

void main() {
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
