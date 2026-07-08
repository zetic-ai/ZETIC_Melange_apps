# Keep the ZeticMLange SDK classes — the native engine references them via JNI, so
# stripping/renaming them causes a SIGABRT at model load. Defensive for a future
# minified build (R8 is off in v1).
-keep class com.zeticai.** { *; }
-dontwarn com.zeticai.**
