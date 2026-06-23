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
        // ZeticMLange Android SDK (`com.zeticai.mlange:mlange`) is published to Maven Central.
        // If the pinned 1.6.1 ever fails to resolve from Central, uncomment and set the
        // Zetic-hosted Maven repo URL (see https://docs.zetic.ai/app_implementation/android.html):
        // maven { url = uri("<ZETIC_MAVEN_REPO_URL>") }
    }
}

rootProject.name = "OfflineTranslator"
include(":app")
