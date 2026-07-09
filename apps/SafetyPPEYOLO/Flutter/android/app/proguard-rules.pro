# Melange native bridge resolves these Kotlin classes at runtime via JNI
# FindClass; R8 cannot see those references and would strip them, causing a
# ClassNotFoundException -> SIGABRT crash-loop at launch. Keep them (protective
# even though minify is disabled for this demo).
-keep class com.zeticai.mlange.** { *; }
-keep class ai.zetic.** { *; }

# Keep native method signatures.
-keepclasseswithmembernames class * {
    native <methods>;
}
