# Melange (zetic_mlange) native bridge resolves its Kotlin classes at runtime
# via JNI FindClass. R8 cannot see those references, so without these keeps it
# strips the classes from the DEX and JNI_OnLoad aborts on launch. Minification
# is currently disabled, but keep these as protection if it is ever re-enabled.
-keep class com.zeticai.mlange.** { *; }
-keep class ai.zetic.** { *; }

# Preserve all native method signatures.
-keepclasseswithmembernames class * {
    native <methods>;
}
