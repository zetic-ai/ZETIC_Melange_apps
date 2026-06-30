plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.google.devtools.ksp")
}

android {
    namespace = "com.brew"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.brew"
        minSdk = 31 // required floor: com.zeticai.mlange:runtimes declares minSdk 31
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        // Universal APK: no abiFilters. The ZeticMLange native engine is arm64-only, so on
        // x86_64 emulators the models will not load (UI stays navigable, transcription/AI no-op).
    }

    buildTypes {
        release {
            // R8 off for v1. The keep rules in proguard-rules.pro are defensive so a future
            // minified build won't strip the SDK's JNI-referenced classes (SIGABRT otherwise).
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    buildFeatures {
        compose = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    // Required by ZeticMLange so its bundled .so libraries are extracted and loadable at runtime
    // (otherwise UnsatisfiedLinkError). pickFirsts resolves the duplicate libc++_shared.so that
    // both native model families ship across all four ABIs.
    packaging {
        jniLibs {
            useLegacyPackaging = true
            pickFirsts += listOf(
                "lib/arm64-v8a/libc++_shared.so",
                "lib/armeabi-v7a/libc++_shared.so",
                "lib/x86/libc++_shared.so",
                "lib/x86_64/libc++_shared.so"
            )
        }
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.12.01")
    implementation(composeBom)

    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.navigation:navigation-compose:2.8.5")

    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    // Room (local persistence for notes + chat messages).
    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    ksp("androidx.room:room-compiler:2.6.1")

    // ZETIC.ai Melange on-device LLM + ASR SDK (Gemma LLM + Whisper encoder/decoder).
    implementation("com.zeticai.mlange:mlange:1.6.1")

    debugImplementation("androidx.compose.ui:ui-tooling")

    testImplementation("junit:junit:4.13.2")
}
