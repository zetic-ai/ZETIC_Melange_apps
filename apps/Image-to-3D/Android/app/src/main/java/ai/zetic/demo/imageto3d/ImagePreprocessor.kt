package ai.zetic.demo.imageto3d

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix

/**
 * Turns a photo into the model's input buffer: center-crop to square
 * (no stretching), resize to 518×518, CHW planar RGB floats in [0, 1].
 * ImageNet normalization is baked into the exported model graph.
 */
object ImagePreprocessor {
    class Output(val chw: FloatArray, val texture: Bitmap)

    fun prepare(source: Bitmap): Output {
        val side = AppConfig.INPUT_SIZE

        // Aspect-fill draw = center crop without stretching.
        val cropped = Bitmap.createBitmap(side, side, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(cropped)
        val scale = maxOf(side.toFloat() / source.width, side.toFloat() / source.height)
        val matrix = Matrix().apply {
            postScale(scale, scale)
            postTranslate((side - source.width * scale) / 2f,
                          (side - source.height * scale) / 2f)
        }
        canvas.drawBitmap(source, matrix, null)

        val area = side * side
        val pixels = IntArray(area)
        cropped.getPixels(pixels, 0, side, 0, 0, side, side)

        val chw = FloatArray(3 * area)
        for (i in 0 until area) {
            val p = pixels[i]
            chw[i] = ((p shr 16) and 0xFF) / 255f            // R plane
            chw[area + i] = ((p shr 8) and 0xFF) / 255f      // G plane
            chw[2 * area + i] = (p and 0xFF) / 255f          // B plane
        }
        return Output(chw, cropped)
    }
}
