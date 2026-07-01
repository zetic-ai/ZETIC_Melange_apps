plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "ai.zetic.demo.cherrypad"
    compileSdk = 35

    defaultConfig {
        applicationId = "ai.zetic.demo.cherrypad"
        minSdk = 31 // required floor: com.zeticai.mlange:runtimes declares minSdk 31
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        // Universal APK: no abiFilters. The ZeticMLange native engine is arm64-only, so on
        // x86_64 emulators the model will not load (the keyboard still types; AI actions no-op).
    }

    buildTypes {
        release {
            // R8 off for v1; keep rules in proguard-rules.pro are defensive against a future
            // minified build stripping the SDK's JNI-referenced classes (SIGABRT otherwise).
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
    // ships across all four ABIs.
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

    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    // Provide the View.setViewTree*Owner extensions used to host Compose inside the IME.
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.7")
    // SavedStateRegistry + ViewTree owners used to host Compose inside the IME service.
    implementation("androidx.savedstate:savedstate:1.2.1")
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    // ZETIC.ai Melange on-device LLM SDK (LFM2.5-350M).
    implementation("com.zeticai.mlange:mlange:1.6.1")

    debugImplementation("androidx.compose.ui:ui-tooling")

    testImplementation("junit:junit:4.13.2")
}
