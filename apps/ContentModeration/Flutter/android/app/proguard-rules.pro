# Keep the ZETIC Melange classes. The native bridge
# (libzetic_mlange_flutter_bridge.so) resolves these Kotlin classes at runtime
# via JNI FindClass, which R8 cannot see — stripping them causes a
# ClassNotFoundException -> JNI_OnLoad abort (SIGABRT) on launch.
# (CLAUDE.md section 5 / LESSONS.md, Android release Bug 1.)
-keep class com.zeticai.mlange.** { *; }
-keep class ai.zetic.** { *; }

# Protective if minify is ever re-enabled: keep native method signatures.
-keepclasseswithmembernames class * {
    native <methods>;
}
