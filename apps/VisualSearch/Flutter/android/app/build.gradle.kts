plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.zeticai.snapseek"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.zeticai.snapseek"
        // ZETIC Melange requires Android API 24+.
        minSdk = maxOf(24, flutter.minSdkVersion)
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            // Signing with the debug keys so `flutter build/run --release` works
            // without a manual keystore for this demo.
            signingConfig = signingConfigs.getByName("debug")

            // LESSONS.md Bug 1: R8 strips the Melange Kotlin classes (e.g.
            // com.zeticai.mlange.core.tensor.Tensor) because they're only
            // referenced from native code via JNI FindClass, which R8 can't
            // see → ClassNotFoundException → SIGABRT crash-loop at launch.
            // Disable shrinking for this demo.
            isMinifyEnabled = false
            isShrinkResources = false
        }
    }

    // Melange ships prebuilt .so libraries; legacy packaging keeps them
    // extractable so the loader can find them at runtime (LESSONS.md).
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
