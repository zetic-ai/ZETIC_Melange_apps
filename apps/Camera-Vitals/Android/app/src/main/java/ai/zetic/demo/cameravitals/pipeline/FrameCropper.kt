package ai.zetic.demo.cameravitals.pipeline

import android.graphics.Bitmap
import android.graphics.Rect
import ai.zetic.demo.cameravitals.AppConfig
import ai.zetic.demo.cameravitals.util.clamped

class CroppedFrame(val planarRGB: FloatArray, val meanLuma: Float)

/**
 * Crops a face ROI from an upright RGB bitmap and resizes to 72×72, emitting planar RGB float
 * (CHW: R plane, G plane, B plane) — exactly what the model input tensor needs.
 */
object FrameCropper {
    private val size = AppConfig.IMG_SIZE

    fun crop(src: Bitmap, roi: Rect): CroppedFrame? {
        val w = src.width
        val h = src.height
        val x = roi.left.clamped(0, w - 1)
        val y = roi.top.clamped(0, h - 1)
        val rw = roi.width().clamped(1, w - x)
        val rh = roi.height().clamped(1, h - y)

        val cropped = Bitmap.createBitmap(src, x, y, rw, rh)
        val scaled = Bitmap.createScaledBitmap(cropped, size, size, true)
        if (cropped !== scaled) cropped.recycle()

        val pixels = IntArray(size * size)
        scaled.getPixels(pixels, 0, size, 0, 0, size, size)
        scaled.recycle()

        val n = size * size
        val planar = FloatArray(3 * n)
        var lumaSum = 0f
        for (i in 0 until n) {
            val p = pixels[i]
            val r = ((p shr 16) and 0xFF).toFloat()
            val g = ((p shr 8) and 0xFF).toFloat()
            val b = (p and 0xFF).toFloat()
            planar[i] = r
            planar[n + i] = g
            planar[2 * n + i] = b
            lumaSum += 0.299f * r + 0.587f * g + 0.114f * b
        }
        return CroppedFrame(planar, lumaSum / n)
    }
}
