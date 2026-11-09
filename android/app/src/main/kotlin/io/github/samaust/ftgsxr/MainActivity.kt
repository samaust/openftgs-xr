// SPDX-License-Identifier: Apache-2.0

package io.github.samaust.ftgsxr

import android.content.Intent
import android.os.Bundle
import com.google.androidgamesdk.GameActivity

/**
 * The Quest app's activity (§2.5, D-18): GameActivity, which loads `libftgsxr_quest.so`
 * (the manifest's `android.app.lib_name`) and runs its `android_main`. Its result callbacks
 * fill the shim's maps (`Shim.kt`), which Rust polls; nothing here calls into Rust.
 */
class MainActivity : GameActivity() {
    companion object {
        init {
            System.loadLibrary("ftgsxr_quest")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        Shim.attach(this)
        super.onCreate(savedInstanceState)
    }

    override fun onPause() {
        // Q-386: the camera closes on pause; Rust starts it again once the session runs.
        Shim.onPause()
        super.onPause()
    }

    override fun onDestroy() {
        Shim.detach(this)
        super.onDestroy()
    }

    @Deprecated("the shim's pickers use startActivityForResult")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (!Shim.onPickResult(this, requestCode, resultCode, data)) {
            @Suppress("DEPRECATION")
            super.onActivityResult(requestCode, resultCode, data)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        Shim.onPermissionResult(requestCode, grantResults)
    }
}
