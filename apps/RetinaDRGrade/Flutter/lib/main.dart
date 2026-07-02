import 'package:flutter/material.dart';

import 'screens/loading_screen.dart';
import 'theme.dart';

void main() {
  runApp(const GradeVueApp());
}

class GradeVueApp extends StatelessWidget {
  const GradeVueApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'GradeVue',
      debugShowCheckedModeBanner: false,
      theme: GradeVueTheme.build(),
      home: const LoadingScreen(),
    );
  }
}
