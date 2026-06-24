// Top-level build file. AGP 8.6.1 / Gradle 8.9. Kotlin 2.2.20 is required because
// ML Kit GenAI Speech Recognition (genai-speech-recognition / genai-common) ships
// metadata compiled with Kotlin 2.2 — older compilers reject it.
plugins {
    id("com.android.application") version "8.6.1" apply false
    id("org.jetbrains.kotlin.android") version "2.2.20" apply false
    // Kotlin's bundled Compose compiler plugin (version-locked to the Kotlin version).
    id("org.jetbrains.kotlin.plugin.compose") version "2.2.20" apply false
}
