// SPDX-License-Identifier: Apache-2.0

package io.github.samaust.ftgsxr

import com.google.androidgamesdk.GameActivity

/**
 * The Quest app's activity (§2.5, D-18): GameActivity, which loads `libftgsxr_quest.so`
 * (the manifest's `android.app.lib_name`) and runs its `android_main`. Nothing here
 * calls into Rust; the shim's methods (`Shim.kt`) join with their milestones.
 */
class MainActivity : GameActivity() {
    companion object {
        init {
            System.loadLibrary("ftgsxr_quest")
        }
    }
}
