# ZETIC Melange keep rules.
#
# The Melange native bridge (libzetic_mlange_flutter_bridge.so) resolves its
# Kotlin/Java classes from native code at runtime via JNI FindClass. R8 cannot
# see those references, so with shrinking enabled it strips the classes and the
# app aborts (ClassNotFoundException: com.zeticai.mlange.core.tensor.Tensor ->
# SIGABRT in JNI_OnLoad) on launch.
#
# Release shrinking is currently disabled (see app/build.gradle.kts), so these
# rules are inert today. They are kept as a safety net: if minify/shrink is ever
# turned on, they preserve everything the native bridge needs.
-keep class com.zeticai.mlange.** { *; }
-keep class ai.zetic.** { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}
