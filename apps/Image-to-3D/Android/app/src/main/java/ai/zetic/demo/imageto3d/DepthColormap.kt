package ai.zetic.demo.imageto3d

import android.graphics.Bitmap

/** Turbo colormap (degree-5 polynomial approximation). Warm = near, cool = far. */
object DepthColormap {
    fun turbo(t: Float): Triple<Float, Float, Float> {
        val x = t.coerceIn(0f, 1f)
        val r = 0.13572138f + x * (4.61539260f + x * (-42.66032258f + x * (132.13108234f + x * (-152.94239396f + x * 59.28637943f))))
        val g = 0.09140261f + x * (2.19418839f + x * (4.84296658f + x * (-14.18503333f + x * (4.27729857f + x * 2.82956604f))))
        val b = 0.10667330f + x * (12.64194608f + x * (-60.58204836f + x * (110.36276771f + x * (-89.90310912f + x * 27.34824973f))))
        return Triple(r.coerceIn(0f, 1f), g.coerceIn(0f, 1f), b.coerceIn(0f, 1f))
    }

    fun bitmap(depth: DepthMap): Bitmap {
        val pixels = IntArray(depth.width * depth.height)
        for (i in pixels.indices) {
            val (r, g, b) = turbo(depth.normalized(i))
            pixels[i] = (0xFF shl 24) or
                ((r * 255).toInt() shl 16) or
                ((g * 255).toInt() shl 8) or
                (b * 255).toInt()
        }
        return Bitmap.createBitmap(pixels, depth.width, depth.height, Bitmap.Config.ARGB_8888)
    }
}
