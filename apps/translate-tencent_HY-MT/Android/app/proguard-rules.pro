# Keep the ZeticMLange SDK and its JNI entry points intact (native methods + classes
# referenced from native code must not be renamed/stripped).
-keep class com.zeticai.** { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}
