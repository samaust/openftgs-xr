// SPDX-License-Identifier: Apache-2.0

package io.github.samaust.ftgsxr

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.DocumentsContract
import android.util.Log
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/**
 * The shim (§2.5, D-49): the static methods Rust calls through JNI, and nothing else of
 * that surface. Kotlin never calls into Rust: a pick's URIs and a permission's answer wait
 * in this object's maps, filled by `MainActivity`'s result callbacks, until Rust polls
 * them. The camera methods are M-7's. The `internal` methods (`attach`, `detach`,
 * `onPickResult`, `onPermissionResult`) are not that surface: they are `MainActivity`'s
 * hooks, neither static nor reachable by name from JNI.
 */
// Q-T452.10
object Shim {
    private const val TAG = "ftgsxr"

    /** The activity the pickers and permission dialogs start from, set by `MainActivity`. */
    @Volatile
    private var activity: Activity? = null

    /** The picks Rust started and not yet polled to completion: null while pending. */
    private val picks = ConcurrentHashMap<Int, Array<String>>()
    private val pendingPicks: MutableSet<Int> = ConcurrentHashMap.newKeySet()
    /** The permission answers by request id: -1 pending, 0 denied, 1 granted. */
    private val permissions = ConcurrentHashMap<Int, Int>()

    /**
     * Android's request codes, kept small and positive (`ActivityCompat.requestPermissions`
     * asks for a code of at least 0; no documentation page gives an upper bound), mapped to
     * Rust's request ids, so the ids may take any value.
     */
    private val nextCode = AtomicInteger(0)
    private val pickCodes = ConcurrentHashMap<Int, Int>()
    private val permissionCodes = ConcurrentHashMap<Int, Int>()

    private fun code(): Int = 1 + (nextCode.getAndIncrement() and 0x7FFF)

    internal fun attach(a: Activity) {
        activity = a
    }

    internal fun detach(a: Activity) {
        if (activity === a) activity = null
    }

    /**
     * Starts `ACTION_OPEN_DOCUMENT` (several files) or `ACTION_OPEN_DOCUMENT_TREE`; the
     * result takes persistable read permission (§2.5).
     */
    @JvmStatic
    fun pickDocuments(requestId: Int, tree: Boolean) {
        val a = activity
        if (a == null) {
            picks[requestId] = emptyArray()
            return
        }
        pendingPicks.add(requestId)
        val code = code()
        pickCodes[code] = requestId
        val intent = if (tree) {
            Intent(Intent.ACTION_OPEN_DOCUMENT_TREE)
        } else {
            Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "*/*"
                putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
            }
        }
        intent.addFlags(
            Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION
        )
        a.runOnUiThread {
            try {
                @Suppress("DEPRECATION")
                a.startActivityForResult(intent, code)
            } catch (e: ActivityNotFoundException) {
                Log.w(TAG, "no document picker: $e")
                pickCodes.remove(code)
                finishPick(requestId, emptyArray())
            }
        }
    }

    /** The content URIs once the picker returns; null while pending; empty when cancelled. */
    @JvmStatic
    fun pollPick(requestId: Int): Array<String>? {
        picks.remove(requestId)?.let { return it }
        return if (pendingPicks.contains(requestId)) null else emptyArray()
    }

    /**
     * The documents of a tree (a picked tree's URI, or a directory's document URI built on
     * it), one `name\tsize\tmime\turi` per entry; each `uri` is built on the tree's, so a
     * directory's lists its children. A failure (a grant revoked) is thrown to the caller.
     */
    // Q-297
    @JvmStatic
    fun listTree(treeUri: String): Array<String> {
        val a = activity ?: throw IllegalStateException("no activity")
        val uri = Uri.parse(treeUri)
        val parent = if (DocumentsContract.isDocumentUri(a, uri)) {
            DocumentsContract.getDocumentId(uri)
        } else {
            DocumentsContract.getTreeDocumentId(uri)
        }
        val children = DocumentsContract.buildChildDocumentsUriUsingTree(uri, parent)
        val columns = arrayOf(
            DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_SIZE,
            DocumentsContract.Document.COLUMN_MIME_TYPE,
        )
        val rows = ArrayList<String>()
        a.contentResolver.query(children, columns, null, null, null)?.use { c ->
            while (c.moveToNext()) {
                val id = c.getString(0) ?: continue
                val name = (c.getString(1) ?: continue).replace('\t', ' ').replace('\n', ' ')
                val size = if (c.isNull(2)) "" else c.getLong(2).toString()
                val mime = c.getString(3) ?: ""
                val child = DocumentsContract.buildDocumentUriUsingTree(uri, id)
                rows.add("$name\t$size\t$mime\t$child")
            }
        } ?: throw IllegalStateException("no listing for $treeUri")
        return rows.toTypedArray()
    }

    /** A detached read-only file descriptor for the URI, -1 on failure. */
    @JvmStatic
    fun openDocument(uri: String): Int {
        val a = activity ?: return -1
        return try {
            a.contentResolver.openFileDescriptor(Uri.parse(uri), "r")?.detachFd() ?: -1
        } catch (e: Exception) {
            Log.w(TAG, "openDocument $uri: $e")
            -1
        }
    }

    /**
     * The runtime permission flow: answered granted at once when the permission is held
     * already, else Android's dialog (Q-276).
     */
    @JvmStatic
    fun requestPermission(requestId: Int, permission: String) {
        val a = activity
        if (a == null) {
            permissions[requestId] = 0
            return
        }
        if (a.checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED) {
            permissions[requestId] = 1
            return
        }
        permissions[requestId] = -1
        val code = code()
        permissionCodes[code] = requestId
        a.runOnUiThread { a.requestPermissions(arrayOf(permission), code) }
    }

    /** -1 pending, 0 denied, 1 granted (an unknown request is denied). */
    @JvmStatic
    fun pollPermission(requestId: Int): Int = permissions[requestId] ?: 0

    /** `MainActivity.onActivityResult`'s share; false when the code is not a pick's. */
    internal fun onPickResult(a: Activity, requestCode: Int, resultCode: Int, data: Intent?): Boolean {
        val requestId = pickCodes.remove(requestCode) ?: return false
        val uris = ArrayList<Uri>()
        if (resultCode == Activity.RESULT_OK && data != null) {
            val clip = data.clipData
            if (clip != null) {
                for (i in 0 until clip.itemCount) clip.getItemAt(i).uri?.let { uris.add(it) }
            } else {
                data.data?.let { uris.add(it) }
            }
        }
        for (uri in uris) {
            try {
                a.contentResolver.takePersistableUriPermission(
                    uri, Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            } catch (e: SecurityException) {
                Log.w(TAG, "no persistable permission for $uri: $e")
            }
        }
        finishPick(requestId, uris.map { it.toString() }.toTypedArray())
        return true
    }

    /** `MainActivity.onRequestPermissionsResult`'s share; false when not a request's. */
    internal fun onPermissionResult(requestCode: Int, grantResults: IntArray): Boolean {
        val requestId = permissionCodes.remove(requestCode) ?: return false
        val granted = grantResults.isNotEmpty() &&
            grantResults.all { it == PackageManager.PERMISSION_GRANTED }
        permissions[requestId] = if (granted) 1 else 0
        return true
    }

    private fun finishPick(requestId: Int, uris: Array<String>) {
        picks[requestId] = uris
        pendingPicks.remove(requestId)
    }
}
