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
    // AGP pinned below the 9.0 line: zetic_mlange 1.8.1 uses the legacy
    // `android {}` + `kotlinOptions {}` DSL, which AGP 9.0 / Kotlin 2.3 turn into
    // hard compile errors. AGP 8.9.1 / Kotlin 2.1.0 build the plugin cleanly.
    // Revert to the Flutter-template defaults once zetic_mlange migrates DSLs.
    id("com.android.application") version "8.9.1" apply false
    id("org.jetbrains.kotlin.android") version "2.1.0" apply false
}

include(":app")
