// SPDX-License-Identifier: Apache-2.0
// The plugin versions (§2.3: the versions `android/` pins). AGP 8.7 runs on Gradle 8.9
// (the wrapper) and JDK 17 or 21, and supports compileSdk 34; Kotlin 2.0.

plugins {
    id("com.android.application") version "8.7.3" apply false
    id("org.jetbrains.kotlin.android") version "2.0.21" apply false
}
