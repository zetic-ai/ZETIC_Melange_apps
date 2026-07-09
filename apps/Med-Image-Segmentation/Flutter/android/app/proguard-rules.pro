# ZeticMLange loads several classes from native code via JNI (FindClass), so R8's
# reachability analysis can't see the references and strips them. The native bridge
# libzetic_mlange_flutter_bridge.so resolves com.zeticai.mlange.core.tensor.Tensor in
# JNI_OnLoad; if R8 removed it the app aborts (SIGABRT, ClassNotFoundException) on
# launch. Flutter force-enables R8 minification for release builds, so we must keep
# every ZeticMLange class (core + all backend runtimes: qnn, tflite, ort, executorch).
-keep class com.zeticai.** { *; }
-keepclassmembers class com.zeticai.** { *; }

# Keep all JNI entry points / native method holders generally.
-keepclasseswithmembernames class * {
    native <methods>;
}
