package ai.zetic.demo.posemotion.video

import android.graphics.Bitmap
import android.media.Image
import java.util.concurrent.Callable
import java.util.concurrent.Executors

/**
 * YUV_420_888 (from MediaCodec.getOutputImage) → ARGB Bitmap.
 * Stride/pixelStride/cropRect aware; integer BT.601 math; luma rows are split
 * across a small fixed pool (synchronous fork-join — deterministic per frame).
 * All scratch buffers are reused; API confined to the inference thread.
 */
class YuvToRgb {
    private var argb = IntArray(0)
    private var yBytes = ByteArray(0)
    private var uBytes = ByteArray(0)
    private var vBytes = ByteArray(0)

    private val workers = Runtime.getRuntime().availableProcessors().coerceIn(2, 4)
    private val pool = Executors.newFixedThreadPool(workers) { r ->
        Thread(r, "yuv2rgb").apply { isDaemon = true }
    }

    fun convert(image: Image, out: Bitmap) {
        val crop = image.cropRect
        val w = crop.width()
        val h = crop.height()

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        // Bulk-copy the planes once (buffer random access from multiple threads is
        // both slow and unsafe; arrays are neither).
        yBytes = copyBuffer(yPlane.buffer, yBytes)
        uBytes = copyBuffer(uPlane.buffer, uBytes)
        vBytes = copyBuffer(vPlane.buffer, vBytes)
        if (argb.size < w * h) argb = IntArray(w * h)

        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        val uRowStride = uPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride
        val vPixelStride = vPlane.pixelStride
        val left = crop.left
        val top = crop.top

        val band = (h + workers - 1) / workers
        val tasks = (0 until workers).mapNotNull { worker ->
            val j0 = worker * band
            val j1 = minOf(h, j0 + band)
            if (j0 >= j1) return@mapNotNull null
            Callable {
                for (j in j0 until j1) {
                    val yRow = (top + j) * yRowStride
                    val chromaRow = (top + j) / 2
                    val uRow = chromaRow * uRowStride
                    val vRow = chromaRow * vRowStride
                    var outIdx = j * w
                    for (i in 0 until w) {
                        val y = (yBytes[yRow + (left + i) * yPixelStride].toInt() and 0xFF) - 16
                        val ci = (left + i) / 2
                        val u = (uBytes[uRow + ci * uPixelStride].toInt() and 0xFF) - 128
                        val v = (vBytes[vRow + ci * vPixelStride].toInt() and 0xFF) - 128

                        // BT.601 limited range, fixed-point ×1024
                        val y1192 = 1192 * y
                        var r = (y1192 + 1634 * v) shr 10
                        var g = (y1192 - 833 * v - 400 * u) shr 10
                        var b = (y1192 + 2066 * u) shr 10
                        if (r < 0) r = 0 else if (r > 255) r = 255
                        if (g < 0) g = 0 else if (g > 255) g = 255
                        if (b < 0) b = 0 else if (b > 255) b = 255

                        argb[outIdx + i] = -0x1000000 or (r shl 16) or (g shl 8) or b
                    }
                }
            }
        }
        pool.invokeAll(tasks).forEach { it.get() }   // rethrows worker exceptions

        out.setPixels(argb, 0, w, 0, 0, w, h)
    }

    private fun copyBuffer(buffer: java.nio.ByteBuffer, reuse: ByteArray): ByteArray {
        buffer.rewind()
        val n = buffer.remaining()
        val arr = if (reuse.size >= n) reuse else ByteArray(n)
        buffer.get(arr, 0, n)
        return arr
    }
}
