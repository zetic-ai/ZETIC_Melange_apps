# Keep the ZeticMLange SDK and its JNI entry points intact (native methods + classes
# referenced from native code via FindClass in JNI_OnLoad must not be renamed/stripped,
# otherwise the native engine SIGABRTs on launch).
-keep class com.zeticai.** { *; }
-keepclassmembers class com.zeticai.** { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}
