package ai.zetic.demo.posemotion.pipeline

import ai.zetic.demo.posemotion.AppConfig
import android.graphics.RectF

/**
 * Person box → pose-model crop rect: pad, then expand to the model's 3:4 (w:h)
 * aspect around the box center, clamped to the frame. Returns frame-pixel coords.
 */
object PersonCropper {
    fun cropRect(personBox: RectF, frameW: Float, frameH: Float): RectF {
        val aspect = AppConfig.POSE_INPUT_WIDTH.toFloat() / AppConfig.POSE_INPUT_HEIGHT

        val boxW = personBox.width() * frameW * AppConfig.CROP_PADDING
        val boxH = personBox.height() * frameH * AppConfig.CROP_PADDING
        val cx = personBox.centerX() * frameW
        val cy = personBox.centerY() * frameH

        // Fit to 3:4 by growing the short side.
        var w = boxW
        var h = boxH
        if (w / h > aspect) h = w / aspect else w = h * aspect

        var left = cx - w / 2
        var top = cy - h / 2

        // Clamp: shift inside the frame first, then intersect as a last resort.
        if (left < 0f) left = 0f
        if (top < 0f) top = 0f
        if (left + w > frameW) left = frameW - w
        if (top + h > frameH) top = frameH - h
        val rect = RectF(left, top, left + w, top + h)
        rect.intersect(0f, 0f, frameW, frameH)
        return rect
    }
}
