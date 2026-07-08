plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.zeticai.pyroguard"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.zeticai.pyroguard"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        // ZETIC Melange requires Android API 24+.
        minSdk = maxOf(24, flutter.minSdkVersion)
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")

            // R8 strips the Melange Kotlin classes (e.g.
            // com.zeticai.mlange.core.tensor.Tensor) because they're only
            // referenced from native code via JNI FindClass, which R8 can't see.
            // That causes a ClassNotFoundException -> SIGABRT crash-loop at launch
            // (JNI_OnLoad). The plugin's bundled consumer ProGuard rules don't
            // cover everything, so disable shrinking for this demo. To ship a
            // minified release instead, set these true and add:
            //   -keep class com.zeticai.mlange.** { *; }
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
