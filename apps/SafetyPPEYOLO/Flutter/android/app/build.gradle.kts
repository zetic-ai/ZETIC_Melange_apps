plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.zeticai.siteguard"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        applicationId = "com.zeticai.siteguard"
        // ZETIC Melange requires Android API 24+.
        minSdk = maxOf(24, flutter.minSdkVersion)
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            // Signing with the debug keys so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")

            // R8 strips the Melange Kotlin classes (referenced only from native
            // code via JNI FindClass, invisible to R8), causing a
            // ClassNotFoundException -> SIGABRT crash-loop at launch
            // (PyroGuard-verified). Disable shrinking for this demo; to minify
            // instead add: -keep class com.zeticai.mlange.** { *; }
            isMinifyEnabled = false
            isShrinkResources = false
        }
    }

    // The Melange native runtime ships prebuilt .so libraries; legacy packaging
    // keeps them extractable so the loader can find them at runtime.
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}
