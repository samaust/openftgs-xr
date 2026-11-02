// SPDX-License-Identifier: Apache-2.0
// The Quest app's Gradle project (§2.5, D-18): one module, `app`, which packages
// `libftgsxr_quest.so` (built by cargo-ndk, `tools/build-quest.sh`).

pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "ftgsxr"
include(":app")
