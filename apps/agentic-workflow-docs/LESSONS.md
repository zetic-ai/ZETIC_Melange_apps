# LESSONS.md — Field-hardened fixes (things that only break on a real device)

A running log of concrete, verified fixes for failures that do not show up until an app runs on physical hardware. It complements CLAUDE.md section 5 ("Hard-won SDK and platform realities"): section 5 is the standing constraint set, this file is the append-only field log. Newest lessons are appended at the bottom.

---

## Android release builds — the two-bug gauntlet (every Flutter + Melange app)

BOTH bugs below are invisible in debug and profile. They only surface in a physical-device RELEASE build. A fresh app scaffolded on the current Flutter/AGP template (AGP 9.0.1 / Kotlin 2.3.20 / Gradle 9.1.0) hits BOTH — treat them as a guaranteed pair, not an edge case. The reference app that already carries all of these fixes is `apps/FireDetectionYOLO` (PyroGuard); copy from it rather than rediscovering.

### Bug 1 — SIGABRT on launch ("splash logo then instant close")

- **Symptom:** the release APK opens to the launcher/splash, then closes immediately.
- **Root cause:** R8 (release code-shrinking) strips `com.zeticai.mlange.core.tensor.Tensor`. The Melange native bridge `libzetic_mlange_flutter_bridge.so` resolves its Kotlin classes at runtime via JNI `FindClass`; R8 cannot see those references, so it removes the class from the DEX → `JNI_OnLoad` → `FindClass` fails → `ClassNotFoundException` → `AssertNoPendingException` → `abort`. The crash buffer (`adb logcat -b crash -d`) shows the `JNI_OnLoad` → `FindClass` → abort chain.
- **Fix (all of it — this is one fix in several files, not a menu):**
  - `android/settings.gradle.kts`: pin `com.android.application` to `8.9.1` and `org.jetbrains.kotlin.android` to `2.1.0` (zetic_mlange 1.8.1's legacy `android{}` / `kotlinOptions{}` DSL hard-errors on AGP 9.x).
  - `android/gradle/wrapper/gradle-wrapper.properties`: Gradle `8.11.1` (9.1.0 is incompatible with AGP 8.9.1 — fails with "FlutterAppPluginLoaderPlugin not found").
  - `android/gradle.properties`: add `android.suppressUnsupportedCompileSdk=36`.
  - `android/app/build.gradle.kts` release buildType: `isMinifyEnabled = false`, `isShrinkResources = false`, wire `proguardFiles(...)`, `packaging { jniLibs { useLegacyPackaging = true } }`, `minSdk = maxOf(24, flutter.minSdkVersion)`.
  - `android/app/proguard-rules.pro` (new): `-keep class com.zeticai.mlange.** { *; }`, `-keep class ai.zetic.** { *; }`, and a keep for native methods (protective if minify is ever re-enabled).
- **Verify on device:** `adb -s <serial> logcat -b crash -c`; `adb shell monkey -p <pkg> 1`; `adb shell pidof <pkg>` returns a PID at t+8s and t+12s; `adb logcat -b crash -d` shows no new abort.

### Bug 2 — "could not load the model"

- **Symptom:** the app launches fine but shows "could not load the model".
- **Root cause:** Flutter injects `android.permission.INTERNET` ONLY into the debug/profile manifest, never the main one. So a RELEASE build has no network access and the Melange model download from S3 fails. `adb shell dumpsys package <pkg> | grep INTERNET` shows it is not granted.
- **Fix:** add to `android/app/src/main/AndroidManifest.xml` (the MAIN manifest, above `<application>`):
  `<uses-permission android:name="android.permission.INTERNET"/>` and `<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE"/>`.
- **Verify on device:** `dumpsys package <pkg> | grep INTERNET` → granted=true; logcat shows `BackendSelectionClient` fetching the S3 URL then `Backend selection success`; the UI reaches its main screen (e.g. ShelfSense showed a live "206 products detected" inference).

### Open issue — model checksum mismatch on large models

- **Symptom:** INTERNET works and the download reaches S3, but the SDK reports a checksum mismatch on the downloaded model file, so it still will not load. Seen on RetinaDRGrade (GradeVue), a ~328 MB fp32 ViT.
- **Under investigation:** transient/partial-download corruption (a clean re-download after `adb shell pm clear <pkg>` may resolve it) vs a stale checksum in the model's server-side registration (a ZETIC backend issue, not client-fixable). If it persists on a clean full re-download, escalate to ZETIC with the exact expected-vs-actual hash + byte counts.

### Where this was fixed

- RetinaDRScreen (FundusGate) PR #66 — crash `46a2f64`, INTERNET `8a9daee`
- RetinaDRGrade (GradeVue) PR #67 — crash `57e0179`, INTERNET `892dc5b`
- ShelfScanYOLO (ShelfSense) PR #64 — crash `58c67fe`, INTERNET `eb28cf1`
- DentalXrayDetect (OraLens) PR #65 — in progress (both fixes)
