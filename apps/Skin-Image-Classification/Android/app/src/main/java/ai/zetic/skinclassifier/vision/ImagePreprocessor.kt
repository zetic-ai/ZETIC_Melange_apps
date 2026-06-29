package ai.zetic.skinclassifier.vision

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import ai.zetic.skinclassifier.core.AppConfig
import ai.zetic.skinclassifier.core.AppConfig.Preprocess.ChannelOrder
import ai.zetic.skinclassifier.core.AppConfig.Preprocess.Layout
import ai.zetic.skinclassifier.core.AppConfig.Preprocess.Normalize

/**
 * Turns an arbitrary photo into the classifier's input tensor values. 1:1 with the iOS
 * `ImagePreprocessor`: stretch-resize to 224×224 (no center crop, matching HF ViTImageProcessor),
 * then pack to NCHW/NHWC float32 with the configured channel order + normalization.
 *
 * Default contract (see [AppConfig.Preprocess]): NCHW `[1,3,224,224]`, RGB, normalized to
 * `[-1,1]` via `px/127.5 - 1`.
 */
object ImagePreprocessor {

    /** Output tensor shape for the configured layout. */
    val shape: IntArray
        get() {
            val s = AppConfig.Preprocess.INPUT_SIZE
            return when (AppConfig.Preprocess.layout) {
                Layout.NCHW -> intArrayOf(1, 3, s, s)
                Layout.NHWC -> intArrayOf(1, s, s, 3)
            }
        }

    /** Build the flat float32 values for [bitmap] following [AppConfig.Preprocess]. */
    fun toInput(bitmap: Bitmap): FloatArray {
        val size = AppConfig.Preprocess.INPUT_SIZE
        val n = size * size

        // 1. Stretch-resize into a 224×224 ARGB_8888 bitmap (no aspect preservation).
        val resized = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(resized)
        val matrix = Matrix().apply { setScale(size.toFloat() / bitmap.width, size.toFloat() / bitmap.height) }
        canvas.drawBitmap(bitmap, matrix, Paint(Paint.FILTER_BITMAP_FLAG))

        val pixels = IntArray(n)
        resized.getPixels(pixels, 0, size, 0, 0, size, size)

        val out = FloatArray(3 * n)
        val nchw = AppConfig.Preprocess.layout == Layout.NCHW
        val bgr = AppConfig.Preprocess.channelOrder == ChannelOrder.BGR

        for (i in 0 until n) {
            val px = pixels[i] // 0xAARRGGBB
            val r = (px ushr 16) and 0xFF
            val g = (px ushr 8) and 0xFF
            val b = px and 0xFF

            // Channel 0..2 in the configured order.
            val c0 = norm(if (bgr) b else r)
            val c1 = norm(g)
            val c2 = norm(if (bgr) r else b)

            if (nchw) {
                out[i] = c0
                out[n + i] = c1
                out[2 * n + i] = c2
            } else {
                out[3 * i] = c0
                out[3 * i + 1] = c1
                out[3 * i + 2] = c2
            }
        }
        resized.recycle()
        return out
    }

    private fun norm(channel: Int): Float = when (AppConfig.Preprocess.normalize) {
        Normalize.SIGNED1 -> channel / 127.5f - 1.0f // [-1, 1]
        Normalize.UNIT -> channel / 255.0f           // [0, 1]
    }
}
