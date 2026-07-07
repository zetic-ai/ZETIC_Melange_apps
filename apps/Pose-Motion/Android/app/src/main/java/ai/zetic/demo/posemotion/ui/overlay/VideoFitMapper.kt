package ai.zetic.demo.posemotion.ui.overlay

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size

/** Aspect-fit math shared by every overlay: where the video frame lands in the canvas. */
object VideoFitMapper {
    fun fitRect(contentW: Float, contentH: Float, container: Size): Rect {
        if (contentW <= 0 || contentH <= 0 || container.width <= 0 || container.height <= 0) {
            return Rect.Zero
        }
        val scale = minOf(container.width / contentW, container.height / contentH)
        val w = contentW * scale
        val h = contentH * scale
        val left = (container.width - w) / 2
        val top = (container.height - h) / 2
        return Rect(left, top, left + w, top + h)
    }

    /** Normalized (0..1) frame point → canvas point. */
    fun point(x: Float, y: Float, fit: Rect): Offset =
        Offset(fit.left + x * fit.width, fit.top + y * fit.height)

    /** Normalized (0..1) frame rect → canvas rect. */
    fun rect(r: android.graphics.RectF, fit: Rect): Rect = Rect(
        fit.left + r.left * fit.width,
        fit.top + r.top * fit.height,
        fit.left + r.right * fit.width,
        fit.top + r.bottom * fit.height,
    )
}
