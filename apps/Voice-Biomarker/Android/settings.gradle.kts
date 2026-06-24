pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // ZeticMLange (com.zeticai.mlange:mlange) is published to Maven Central;
        // JitPack kept as a fallback mirror.
        maven { url = uri("https://jitpack.io") }
    }
}

rootProject.name = "VoiceVitals"
include(":app")
