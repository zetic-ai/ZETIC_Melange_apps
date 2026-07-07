pluginManagement {
    val flutterSdkPath =
        run {
            val properties = java.util.Properties()
            file("local.properties").inputStream().use { properties.load(it) }
            val flutterSdkPath = properties.getProperty("flutter.sdk")
            require(flutterSdkPath != null) { "flutter.sdk not set in local.properties" }
            flutterSdkPath
        }

    includeBuild("$flutterSdkPath/packages/flutter_tools/gradle")

    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins {
    id("dev.flutter.flutter-plugin-loader") version "1.0.0"
    // Pinned: zetic_mlange 1.8.1's android/build.gradle.kts uses the legacy Gradle
    // DSL, which AGP 9.0 (newDsl) and Kotlin >=2.2 reject as hard errors. These
    // versions clear Flutter 3.44.4's floors and match the verified VoxScribe build.
    id("com.android.application") version "8.12.0" apply false
    id("org.jetbrains.kotlin.android") version "2.1.21" apply false
}

include(":app")
