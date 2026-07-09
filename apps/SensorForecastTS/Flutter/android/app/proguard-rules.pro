# Keep the ZETIC Melange JNI bridge classes. The native bridge
# (libzetic_mlange_flutter_bridge.so) resolves these via JNI FindClass at
# runtime, which R8 cannot see — without these keeps R8 strips the class and
# JNI_OnLoad aborts on launch (SIGABRT).
-keep class com.zeticai.mlange.** { *; }
-keep class ai.zetic.** { *; }

# Keep all native method signatures (protective if minify is ever re-enabled).
-keepclasseswithmembernames class * {
    native <methods>;
}
