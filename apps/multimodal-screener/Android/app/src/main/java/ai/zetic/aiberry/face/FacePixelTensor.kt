package ai.zetic.aiberry.face

import android.graphics.Bitmap
import ai.zetic.aiberry.core.AppConfig

/**
 * Turns a 224x224 face crop into the exact tensor `ElenaRyumina/FaceEmotionRecognition`
 * expects: **NCHW, BGR channel order, raw 0-255 with per-channel mean subtraction, no /255,
 * no std** (Caffe/VGGFace2 convention). 1:1 with iOS-Aiberry's `FacePixelTensor.bgrMeanSubtracted`.
 *
 * Output layout (length 3*224*224 = 150528): all Blue values, then all Green, then all Red.
 */
object FacePixelTensor {

    /** [bitmap] must be ARGB_8888 and exactly [AppConfig.Face.INPUT_SIZE] square. */
    fun bgrMeanSubtracted(bitmap: Bitmap): FloatArray {
        val size = AppConfig.Face.INPUT_SIZE
        val n = size * size
        val out = FloatArray(3 * n)
        val pixels = IntArray(n)
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size)
        for (i in 0 until n) {
            val px = pixels[i]              // 0xAARRGGBB
            val r = (px ushr 16) and 0xFF
            val g = (px ushr 8) and 0xFF
            val b = px and 0xFF
            out[i] = b.toFloat() - AppConfig.Face.MEAN_B           // B plane
            out[n + i] = g.toFloat() - AppConfig.Face.MEAN_G       // G plane
            out[2 * n + i] = r.toFloat() - AppConfig.Face.MEAN_R   // R plane
        }
        return out
    }
}
