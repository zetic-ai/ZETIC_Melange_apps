# ZETIC Melange native bridge resolves its Kotlin classes at runtime via JNI
# FindClass, which R8 cannot see. Keep them so a minified release does not crash
# in JNI_OnLoad (LESSONS.md). Minify is currently off, so these are protective.
-keep class com.zeticai.mlange.** { *; }
-keep class ai.zetic.** { *; }

# Keep native methods.
-keepclasseswithmembernames class * {
    native <methods>;
}
