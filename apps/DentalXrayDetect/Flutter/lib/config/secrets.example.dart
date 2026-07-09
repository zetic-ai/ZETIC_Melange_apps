// Template for the (gitignored) real secrets.dart.
//
// Copy this file to `secrets.dart` in the same folder and paste your ZETIC
// personal key. `secrets.dart` is listed in .gitignore and must NEVER be
// committed. The key is embedded in the client at build time (see CLAUDE.md §5,
// Tier C "Secrets").
//
//   cp lib/config/secrets.example.dart lib/config/secrets.dart
//   # then edit secrets.dart and replace the placeholder with your key
//
/// ZETIC Melange personal key. Replace the placeholder in your local
/// `secrets.dart`; do not commit the real value.
const String zeticPersonalKey = '<ZETIC_PERSONAL_KEY>';
