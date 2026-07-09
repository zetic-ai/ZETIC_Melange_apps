# ZETIC Melange keep rules.
#
# The Melange Kotlin classes (e.g. com.zeticai.mlange.core.tensor.Tensor) are
# referenced only from native code via JNI FindClass in JNI_OnLoad, which R8
# cannot see. Without these keeps, a minified release strips them and the app
# aborts with ClassNotFoundException -> SIGABRT at launch.
#
# NOTE: the release buildType disables shrinking (isMinifyEnabled = false), so
# these rules are inert today. They are kept here so that turning minify back on
# for a shipping release stays safe.
-keep class com.zeticai.mlange.** { *; }
-keep class ai.zetic.** { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}
