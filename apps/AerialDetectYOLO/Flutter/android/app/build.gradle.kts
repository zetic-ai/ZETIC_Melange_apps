plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.zetic.aerialdetect"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        applicationId = "com.zetic.aerialdetect"
        // Melange (zetic_mlange) requires minSdk 24.
        minSdk = maxOf(24, flutter.minSdkVersion)
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    // The Melange native .so's are loaded via JNI; legacy (uncompressed)
    // packaging keeps them mappable from the APK.
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    buildTypes {
        release {
            // Signing with debug keys so `flutter run --release` works without a
            // signing identity; replace with a real signing config to ship.
            signingConfig = signingConfigs.getByName("debug")
            // R8 strips the Melange Kotlin classes (referenced only from native
            // code via JNI FindClass), which crashes JNI_OnLoad at launch. Keep
            // minification off (re-enable with -keep class com.zeticai.mlange.**).
            isMinifyEnabled = false
            isShrinkResources = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
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
