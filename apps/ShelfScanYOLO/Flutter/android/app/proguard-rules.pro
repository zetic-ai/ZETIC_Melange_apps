# ZETIC Melange native runtime.
# These Kotlin/Java classes are looked up only from native code via
# JNI FindClass (e.g. com.zeticai.mlange.core.tensor.Tensor in JNI_OnLoad of
# libzetic_mlange_flutter_bridge.so). R8 cannot see those references, so without
# explicit keeps it strips the classes -> ClassNotFoundException -> SIGABRT
# crash-loop at launch. Keep the whole Melange surface as a safety net for
# minified release builds.
-keep class com.zeticai.mlange.** { *; }
-keep class ai.zetic.** { *; }
-keepclasseswithmembers class com.zeticai.mlange.** {
    native <methods>;
}
-dontwarn com.zeticai.mlange.**
-dontwarn ai.zetic.**
