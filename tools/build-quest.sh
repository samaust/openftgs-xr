#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Builds the Quest APK (§2.5): `libftgsxr_quest.so` with cargo-ndk into android/'s
# jniLibs, then Gradle. Usage: tools/build-quest.sh [debug|release] (default debug).
# Prints the APK's path last. Needs the aarch64-linux-android target, cargo-ndk, JDK 17
# or later, and ANDROID_HOME (default ~/Android/Sdk) with NDK 25.1.8937393 and
# platforms;android-34. Extra Gradle arguments come from GRADLE_ARGS (e.g.
# "--max-workers=8"); cargo honours CARGO_BUILD_JOBS.
set -euo pipefail

profile="${1:-debug}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$ANDROID_HOME/ndk/25.1.8937393}"

case "$profile" in
  debug) cargo_flags=(); task=assembleDebug ;;
  release) cargo_flags=(--release); task=assembleRelease ;;
  *) echo "usage: $0 [debug|release]" >&2; exit 2 ;;
esac

cd "$root"
cargo ndk -t arm64-v8a -o android/app/src/main/jniLibs build "${cargo_flags[@]}" -p ftgsxr-quest

cd "$root/android"
# shellcheck disable=SC2086
./gradlew --no-daemon ${GRADLE_ARGS:-} "$task"

if [ "$profile" = debug ]; then
  echo "$root/android/app/build/outputs/apk/debug/app-debug.apk"
else
  echo "$root/android/app/build/outputs/apk/release/app-release-unsigned.apk"
fi
