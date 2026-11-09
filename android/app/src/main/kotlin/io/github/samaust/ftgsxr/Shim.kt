// SPDX-License-Identifier: Apache-2.0

package io.github.samaust.ftgsxr

import android.annotation.SuppressLint
import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.Image
import android.media.ImageReader
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.provider.DocumentsContract
import android.util.Log
import android.util.Range
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/**
 * The shim (§2.5, D-49): the static methods Rust calls through JNI, and nothing else of
 * that surface. Kotlin never calls into Rust: a pick's URIs and a permission's answer wait
 * in this object's maps, filled by `MainActivity`'s result callbacks, until Rust polls
 * them, and a camera thumbnail waits likewise until `pollCameraThumb` takes it. The
 * `internal` methods (`attach`, `detach`, `onPickResult`, `onPermissionResult`, `onPause`)
 * are not that surface: they are `MainActivity`'s
 * hooks, neither static nor reachable by name from JNI.
 */
// Q-319, Q-386
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

    // The passthrough camera (§1.9, §2.5, D-15, Q-386): the left passthrough camera through
    // Camera2, a 1280 × 960 YUV_420_888 ImageReader at 60 Hz on a handler thread of its
    // own, each frame reduced to 64 × 48 RGB. No Gradle dependency: Camera2 and ImageReader
    // are the SDK's.

    private const val CAMERA_WIDTH = 1280
    private const val CAMERA_HEIGHT = 960
    private const val THUMB_WIDTH = 64
    private const val THUMB_HEIGHT = 48
    /** 1280 / 64 = 960 / 48 = 20: each thumbnail pixel is a 20 × 20 block of Y. */
    private const val BLOCK = 20
    private const val CAMERA_SOURCE = "com.meta.extra_metadata.camera_source"
    private const val CAMERA_POSITION = "com.meta.extra_metadata.position"

    private val cameraLock = Any()
    private var cameraThread: HandlerThread? = null
    private var cameraDevice: CameraDevice? = null
    private var cameraSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    /** The lens pose and field of view (11 floats), or null when the characteristics lack them. */
    private var cameraLens: FloatArray? = null
    private var dataSpaceLogged = false
    /** Set while a frame is being reduced: a frame arriving meanwhile is dropped. */
    private val reducing = AtomicBoolean(false)
    @Volatile
    private var thumb: FloatArray? = null

    /**
     * Opens the left passthrough camera: 0 opening (thumbnails follow), 1 no passthrough
     * camera listed, 2 a camera permission missing, 3 the open failed. An error after 0 is
     * logged and closes the camera. Already open, 0 at once.
     */
    @JvmStatic
    @SuppressLint("MissingPermission")
    fun startCamera(): Int {
        val a = activity ?: return 3
        synchronized(cameraLock) {
            if (cameraThread != null) return 0
            for (p in arrayOf("android.permission.CAMERA", "horizonos.permission.HEADSET_CAMERA")) {
                if (a.checkSelfPermission(p) != PackageManager.PERMISSION_GRANTED) return 2
            }
            val manager = a.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val id = try {
                manager.cameraIdList.firstOrNull { isLeftPassthrough(manager.getCameraCharacteristics(it)) }
            } catch (e: Exception) {
                Log.w(TAG, "listing the cameras: $e")
                return 3
            } ?: return 1
            val chars = manager.getCameraCharacteristics(id)
            cameraLens = lens(chars)
            Log.i(TAG, "passthrough camera $id, lens ${cameraLens?.joinToString(",") ?: "unknown"}")
            val thread = HandlerThread("ftgsxr-camera").also { it.start() }
            val handler = Handler(thread.looper)
            cameraThread = thread
            val reader = ImageReader.newInstance(CAMERA_WIDTH, CAMERA_HEIGHT, ImageFormat.YUV_420_888, 2)
            reader.setOnImageAvailableListener({ onImage(it) }, handler)
            imageReader = reader
            try {
                manager.openCamera(id, object : CameraDevice.StateCallback() {
                    override fun onOpened(device: CameraDevice) {
                        synchronized(cameraLock) {
                            if (cameraThread !== thread) {
                                device.close()
                                return
                            }
                            cameraDevice = device
                        }
                        startSession(device, reader, handler, thread)
                    }

                    override fun onDisconnected(device: CameraDevice) {
                        Log.w(TAG, "the passthrough camera was disconnected")
                        closeCamera()
                    }

                    override fun onError(device: CameraDevice, error: Int) {
                        Log.w(TAG, "the passthrough camera failed: error $error")
                        closeCamera()
                    }
                }, handler)
            } catch (e: Exception) {
                Log.w(TAG, "the passthrough camera could not be opened: $e")
                closeLocked()
                return 3
            }
            return 0
        }
    }

    /** The newest thumbnail's array once (Q-386's layout), null until the next. */
    @JvmStatic
    fun pollCameraThumb(): FloatArray? {
        synchronized(cameraLock) {
            val t = thumb
            thumb = null
            return t
        }
    }

    /** Closes the camera; nothing reopens it but `startCamera`. */
    @JvmStatic
    fun stopCamera() {
        closeCamera()
    }

    /** `MainActivity.onPause`'s share: the camera closes, and Rust starts it again. */
    internal fun onPause() {
        closeCamera()
    }

    /** The camera whose Meta metadata names the passthrough source (0) and the left position (0). */
    private fun isLeftPassthrough(c: CameraCharacteristics): Boolean =
        vendorInt(c, CAMERA_SOURCE) == 0 && vendorInt(c, CAMERA_POSITION) == 0

    /** A vendor tag as an integer: the Quest 3 stores these as one byte. */
    private fun vendorInt(c: CameraCharacteristics, name: String): Int? {
        try {
            c.get(CameraCharacteristics.Key(name, Byte::class.javaObjectType))?.let { return it.toInt() }
        } catch (e: Exception) {
            // Not a byte on this OS: try an int.
        }
        return try {
            c.get(CameraCharacteristics.Key(name, Int::class.javaObjectType))
        } catch (e: Exception) {
            null
        }
    }

    /**
     * The lens pose relative to the headset in OpenXR's axes and the stream's field of
     * view, 11 floats, or null without the characteristics. `LENS_POSE_ROTATION` rotates
     * Android's sensor axes (x right, y up, z back on the Quest 3: OpenXR's) into the
     * camera's (x along the long side, y down the short side, z along the optical axis);
     * the camera's orientation looking along −z with y up is its inverse turned by π about
     * x. The intrinsics are in the pre-correction array's pixels; the 1280 × 960 stream is
     * that array's centred crop to 4:3 (`SCALER_CROP_REGION`), against which the four
     * angles are taken.
     */
    private fun lens(c: CameraCharacteristics): FloatArray? {
        val t = c.get(CameraCharacteristics.LENS_POSE_TRANSLATION) ?: return null
        val q = c.get(CameraCharacteristics.LENS_POSE_ROTATION) ?: return null
        val k = c.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION) ?: return null
        val array = c.get(CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE)
            ?: return null
        if (t.size < 3 || q.size < 4 || k.size < 4 || k[0] <= 0f || k[1] <= 0f) return null
        // conj(q) ⊗ (1, 0, 0, 0), as (x, y, z, w).
        val (qx, qy, qz, qw) = listOf(-q[0], -q[1], -q[2], q[3])
        val rx = qw
        val ry = qz
        val rz = -qy
        val rw = -qx
        val w = array.width().toFloat()
        val h = array.height().toFloat()
        val aspect = CAMERA_WIDTH.toFloat() / CAMERA_HEIGHT
        val cw = minOf(w, h * aspect)
        val ch = minOf(h, w / aspect)
        val ox = (w - cw) / 2f
        val oy = (h - ch) / 2f
        val (fx, fy, cx, cy) = listOf(k[0], k[1], k[2] - ox, k[3] - oy)
        return floatArrayOf(
            t[0], t[1], t[2], rx, ry, rz, rw,
            -kotlin.math.atan(cx / fx), kotlin.math.atan((cw - cx) / fx),
            kotlin.math.atan(cy / fy), -kotlin.math.atan((ch - cy) / fy),
        )
    }

    private fun startSession(device: CameraDevice, reader: ImageReader, handler: Handler, thread: HandlerThread) {
        try {
            @Suppress("DEPRECATION")
            device.createCaptureSession(listOf(reader.surface), object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    synchronized(cameraLock) {
                        if (cameraThread !== thread) {
                            session.close()
                            return
                        }
                        cameraSession = session
                    }
                    try {
                        val request = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                            addTarget(reader.surface)
                            set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(60, 60))
                        }.build()
                        session.setRepeatingRequest(request, null, handler)
                    } catch (e: Exception) {
                        Log.w(TAG, "the passthrough camera's request failed: $e")
                        closeCamera()
                    }
                }

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    Log.w(TAG, "the passthrough camera's session could not be configured")
                    closeCamera()
                }
            }, handler)
        } catch (e: Exception) {
            Log.w(TAG, "the passthrough camera's session failed: $e")
            closeCamera()
        }
    }

    /** One frame: the newest image, reduced unless a reduction is running. */
    private fun onImage(reader: ImageReader) {
        val image = try {
            reader.acquireLatestImage()
        } catch (e: IllegalStateException) {
            null
        } ?: return
        try {
            if (!reducing.compareAndSet(false, true)) return
            try {
                logDataSpace(image)
                val out = reduce(image)
                synchronized(cameraLock) { thumb = out }
            } finally {
                reducing.set(false)
            }
        } finally {
            image.close()
        }
    }

    /** The frame's data space once (its range: Q-386's full-range assumption), API 33 on. */
    private fun logDataSpace(image: Image) {
        if (dataSpaceLogged) return
        dataSpaceLogged = true
        if (Build.VERSION.SDK_INT >= 33) {
            val ds = image.dataSpace
            val range = when (android.hardware.DataSpace.getRange(ds)) {
                android.hardware.DataSpace.RANGE_FULL -> "full"
                android.hardware.DataSpace.RANGE_LIMITED -> "limited"
                android.hardware.DataSpace.RANGE_UNSPECIFIED -> "unspecified"
                else -> "other"
            }
            Log.i(TAG, "passthrough camera data space $ds, range $range")
        }
    }

    /**
     * The 64 × 48 thumbnail: the mean of each 20 × 20 Y block and its 10 × 10 Cb and Cr
     * blocks, through BT.709's full-range matrix to RGB in [0, 1]; then the timestamp as
     * whole seconds and fraction, and the lens's 11 floats when known (Q-386).
     */
    private fun reduce(image: Image): FloatArray {
        val planes = image.planes
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]
        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer
        val yStride = yPlane.rowStride
        val cStride = uPlane.rowStride
        val cPixel = uPlane.pixelStride
        val lens = cameraLens
        val out = FloatArray(THUMB_WIDTH * THUMB_HEIGHT * 3 + 2 + (lens?.size ?: 0))
        val ySums = IntArray(THUMB_WIDTH)
        val uSums = IntArray(THUMB_WIDTH)
        val vSums = IntArray(THUMB_WIDTH)
        val yRow = ByteArray(CAMERA_WIDTH)
        val cLen = (CAMERA_WIDTH / 2 - 1) * cPixel + 1
        val uRow = ByteArray(cLen)
        val vRow = ByteArray(cLen)
        val half = BLOCK / 2
        for (by in 0 until THUMB_HEIGHT) {
            ySums.fill(0)
            uSums.fill(0)
            vSums.fill(0)
            for (r in 0 until BLOCK) {
                yBuf.position((by * BLOCK + r) * yStride)
                yBuf.get(yRow, 0, CAMERA_WIDTH)
                for (bx in 0 until THUMB_WIDTH) {
                    var s = 0
                    val x0 = bx * BLOCK
                    for (x in x0 until x0 + BLOCK) s += yRow[x].toInt() and 0xFF
                    ySums[bx] += s
                }
            }
            for (r in 0 until half) {
                val row = (by * half + r) * cStride
                uBuf.position(row)
                uBuf.get(uRow, 0, cLen)
                vBuf.position(row)
                vBuf.get(vRow, 0, cLen)
                for (bx in 0 until THUMB_WIDTH) {
                    var su = 0
                    var sv = 0
                    val x0 = bx * half
                    for (x in x0 until x0 + half) {
                        su += uRow[x * cPixel].toInt() and 0xFF
                        sv += vRow[x * cPixel].toInt() and 0xFF
                    }
                    uSums[bx] += su
                    vSums[bx] += sv
                }
            }
            for (bx in 0 until THUMB_WIDTH) {
                val y = ySums[bx] / (BLOCK * BLOCK * 255f)
                val cb = uSums[bx] / (half * half * 255f) - 0.5f
                val cr = vSums[bx] / (half * half * 255f) - 0.5f
                // BT.709, full range: R = Y + 1.5748 Cr, G = Y − 0.1873 Cb − 0.4681 Cr,
                // B = Y + 1.8556 Cb.
                val i = (by * THUMB_WIDTH + bx) * 3
                out[i] = (y + 1.5748f * cr).coerceIn(0f, 1f)
                out[i + 1] = (y - 0.18732f * cb - 0.46812f * cr).coerceIn(0f, 1f)
                out[i + 2] = (y + 1.8556f * cb).coerceIn(0f, 1f)
            }
        }
        val ns = image.timestamp
        val n = THUMB_WIDTH * THUMB_HEIGHT * 3
        out[n] = (ns / 1_000_000_000L).toFloat()
        out[n + 1] = (ns % 1_000_000_000L) / 1e9f
        lens?.copyInto(out, n + 2)
        return out
    }

    private fun closeCamera() {
        synchronized(cameraLock) { closeLocked() }
    }

    private fun closeLocked() {
        try {
            cameraSession?.close()
        } catch (e: Exception) {
            Log.w(TAG, "closing the camera's session: $e")
        }
        cameraSession = null
        cameraDevice?.close()
        cameraDevice = null
        imageReader?.close()
        imageReader = null
        cameraThread?.quitSafely()
        cameraThread = null
        thumb = null
        dataSpaceLogged = false
    }
}
