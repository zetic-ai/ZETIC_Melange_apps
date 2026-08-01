import 'package:flutter/material.dart';

import 'screens/loading_screen.dart';
import 'theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const RedactLensApp());
}

/// RedactLens — live on-device document PII redaction.
///
/// Two Melange models (PP-OCRv5 DBNet detector + CRNN/SVTR CTC recognizer)
/// run fully on-device; PII fields are redacted in the live preview before
/// anything could be stored or sent.
class RedactLensApp extends StatelessWidget {
  const RedactLensApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'RedactLens',
      debugShowCheckedModeBanner: false,
      theme: buildRedactLensTheme(),
      home: const LoadingScreen(),
    );
  }
}
