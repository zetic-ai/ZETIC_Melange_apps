plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "ai.zetic.demo.offlinetranslator"
    compileSdk = 35

    defaultConfig {
        applicationId = "ai.zetic.demo.offlinetranslator"
        minSdk = 31 // required floor: com.zeticai.mlange:runtimes declares minSdk 31
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        // Universal APK: no abiFilters. The ZeticMLange native engine is arm64-only, so on
        // x86_64 emulators TranslatorFactory falls back to MockTranslator at runtime.
    }

    buildTypes {
        release {
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

    // Required by ZeticMLange so its bundled .so libraries are extracted and loadable at runtime.
    packaging {
        jniLibs {
            useLegacyPackaging = true
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

    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    // ZETIC.ai Melange on-device LLM SDK (Tencent Hunyuan-MT runtime).
    implementation("com.zeticai.mlange:mlange:1.6.1")

    // Offline speech-to-text (ML Kit GenAI, BASIC mode; one-time on-device model download).
    implementation("com.google.mlkit:genai-speech-recognition:1.0.0-alpha1")

    // Offline OCR (ML Kit Text Recognition; bundled models). Latin default + extra scripts.
    implementation("com.google.mlkit:text-recognition:16.0.1")
    implementation("com.google.mlkit:text-recognition-chinese:16.0.1")
    implementation("com.google.mlkit:text-recognition-japanese:16.0.1")
    implementation("com.google.mlkit:text-recognition-korean:16.0.1")
    implementation("com.google.mlkit:text-recognition-devanagari:16.0.1")

    debugImplementation("androidx.compose.ui:ui-tooling")
}
