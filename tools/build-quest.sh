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

# Q-123: the debug build bundles the Khronos validation layer (arm64-v8a) from a
# Vulkan-ValidationLayers Android release (Apache-2.0), downloaded and checked here, never
# committed; the dv driver enables it through Android's GPU debug-layer settings.
vvl_version=1.4.304.1
vvl_sha256=f7b3688442437bc278f81cbbe1d78a701c32e9cdbaabcf136028d6097602fac6
vvl_dir="$root/android/app/src/debug/jniLibs/arm64-v8a"
if [ "$profile" = debug ] && [ ! -f "$vvl_dir/libVkLayer_khronos_validation.so" ]; then
  cache="${XDG_CACHE_HOME:-$HOME/.cache}/ftgsxr"
  zip="$cache/android-binaries-$vvl_version.zip"
  mkdir -p "$cache"
  if [ ! -f "$zip" ]; then
    curl -sSLf -o "$zip.part" \
      "https://github.com/KhronosGroup/Vulkan-ValidationLayers/releases/download/vulkan-sdk-$vvl_version/android-binaries-$vvl_version.zip"
    mv "$zip.part" "$zip"
  fi
  echo "$vvl_sha256  $zip" | sha256sum -c --quiet -
  mkdir -p "$vvl_dir"
  unzip -o -j -q "$zip" "android-binaries-$vvl_version/arm64-v8a/libVkLayer_khronos_validation.so" -d "$vvl_dir"
fi

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
