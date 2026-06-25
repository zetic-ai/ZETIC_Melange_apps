plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "ai.zetic.voicevitals"
    compileSdk = 35

    defaultConfig {
        applicationId = "ai.zetic.voicevitals"
        // ZeticMLange's runtimes AAR (com.zeticai.mlange:runtimes) requires minSdk 31.
        minSdk = 31
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
    }

    // Required by the ZeticMLange runtime so its native (.so) libraries are
    // extracted and loadable on-device.
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

dependencies {
    // ZeticMLange on-device inference SDK. 1.8.1 (not the 1.6.1 in the brief) is
    // required: the emotion model's v2 artifact is published as an ExecuTorch FP32
    // target, and ExecuTorch support (Target.EXECUTORCH_FP32 + the executorch
    // runtime) only exists in core >= 0.1.1, which ships with mlange 1.8.x.
    // (Trade-off: 1.8.x selects the backend online on every load, so a cold start
    // needs connectivity; 1.6.1 selects locally/offline but can't load ExecuTorch.)
    implementation("com.zeticai.mlange:mlange:1.8.1")

    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.2")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.6")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.6")

    val composeBom = platform("androidx.compose:compose-bom:2024.09.03")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")
}
