import 'package:flutter/material.dart';

import 'screens/loading_screen.dart';
import 'theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const SiteGuardApp());
}

class SiteGuardApp extends StatelessWidget {
  const SiteGuardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SiteGuard',
      debugShowCheckedModeBanner: false,
      theme: buildSiteGuardTheme(),
      home: const LoadingScreen(),
    );
  }
}
