// SPDX-License-Identifier: Apache-2.0
// The Quest app (§2.5): `MainActivity : GameActivity` (D-18), the Khronos OpenXR loader
// from Maven (D-19), arm64-v8a only, `libftgsxr_quest.so` from `src/main/jniLibs`
// (cargo-ndk). Debug builds sign with Gradle's default debug keystore (outside the
// repository); release signing is the owner's, so the release APK is unsigned here.

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    // D-36: the owner's application id (2026-11-01).
    namespace = "io.github.samaust.ftgsxr"
    compileSdk = 34
    ndkVersion = "25.1.8937393"

    defaultConfig {
        applicationId = "io.github.samaust.ftgsxr"
        minSdk = 32
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        getByName("release") {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    packaging {
        jniLibs {
            // GameActivity finds the library by name; keep the symbols' layout simple.
            useLegacyPackaging = true
        }
    }
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    // D-18: android-activity 0.6.0's GameActivity glue is 2.0.2's; newer AARs do not
    // match it.
    implementation("androidx.games:games-activity:2.0.2")
    // GameActivity 2.0.2 extends AppCompatActivity, whose classes its POM leaves to the
    // app (both Apache-2.0, part of "the GameActivity library" of §2.3).
    // Q-T221F.3
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.core:core:1.12.0")
    // D-19: ≥ 1.0.34 as Meta requires; 1.1.46 and later also ship `jni/arm64-v8a`, so
    // Gradle packages `libopenxr_loader.so` without a prefab consumer.
    implementation("org.khronos.openxr:openxr_loader_for_android:1.1.65")
}
