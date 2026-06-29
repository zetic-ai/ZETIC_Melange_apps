// Top-level build file. AGP 8.6.1 / Gradle 8.9 / Kotlin 2.2.20 with the bundled
// Compose compiler plugin — the same toolchain the sibling Melange demo apps build with.
plugins {
    id("com.android.application") version "8.6.1" apply false
    id("org.jetbrains.kotlin.android") version "2.2.20" apply false
    // Kotlin's bundled Compose compiler plugin (version-locked to the Kotlin version).
    id("org.jetbrains.kotlin.plugin.compose") version "2.2.20" apply false
}
