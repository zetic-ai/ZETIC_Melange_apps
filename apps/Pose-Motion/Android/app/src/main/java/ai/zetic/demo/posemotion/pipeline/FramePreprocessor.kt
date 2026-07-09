package ai.zetic.demo.posemotion.pipeline

import ai.zetic.demo.posemotion.AppConfig
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import com.zeticai.mlange.core.tensor.DataType
import com.zeticai.mlange.core.tensor.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Bitmap → NCHW float tensors with every buffer preallocated (zero steady-state
 * allocation, pattern from apps/YOLO26/Android). Confined to the inference thread.
 */
class FramePreprocessor {
    private val detSize = AppConfig.DET_SIZE
    private val poseW = AppConfig.POSE_INPUT_WIDTH
    private val poseH = AppConfig.POSE_INPUT_HEIGHT

    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val matrix = Matrix()

    private val detBitmap = Bitmap.createBitmap(detSize, detSize, Bitmap.Config.ARGB_8888)
    private val detCanvas = Canvas(detBitmap)
    private val detPixels = IntArray(detSize * detSize)
    private val detFloats = FloatArray(3 * detSize * detSize)
    private val detBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(detFloats.size * 4).order(ByteOrder.nativeOrder())

    private val poseBitmap = Bitmap.createBitmap(poseW, poseH, Bitmap.Config.ARGB_8888)
    private val poseCanvas = Canvas(poseBitmap)
    private val posePixels = IntArray(poseW * poseH)
    private val poseFloats = FloatArray(3 * poseW * poseH)
    private val poseBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(poseFloats.size * 4).order(ByteOrder.nativeOrder())

    private val srcRect = Rect()
    private val dstRect = Rect()

    /** Full frame → [1,3,640,640] RGB 0..1 (stretch resize; the detector's normalized
     *  outputs map straight back to the frame, so the stretch cancels). */
    fun detectorTensor(frame: Bitmap): Tensor {
        matrix.reset()
        matrix.setScale(detSize.toFloat() / frame.width, detSize.toFloat() / frame.height)
        detCanvas.drawBitmap(frame, matrix, paint)
        return pack(detBitmap, detPixels, detFloats, detBuffer, detSize, detSize)
    }

    /** Person crop (frame pixels) → [1,3,256,192] RGB 0..1. */
    fun poseTensor(frame: Bitmap, cropPx: RectF): Tensor {
        val x = cropPx.left.toInt().coerceIn(0, frame.width - 2)
        val y = cropPx.top.toInt().coerceIn(0, frame.height - 2)
        val w = cropPx.width().toInt().coerceIn(2, frame.width - x)
        val h = cropPx.height().toInt().coerceIn(2, frame.height - y)
        srcRect.set(x, y, x + w, y + h)
        dstRect.set(0, 0, poseW, poseH)
        poseCanvas.drawBitmap(frame, srcRect, dstRect, paint)
        return pack(poseBitmap, posePixels, poseFloats, poseBuffer, poseW, poseH)
    }

    private fun pack(
        bitmap: Bitmap, pixels: IntArray, floats: FloatArray, buffer: ByteBuffer,
        w: Int, h: Int,
    ): Tensor {
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        val area = w * h
        val norm = 1f / 255f
        for (i in 0 until area) {
            val p = pixels[i]
            floats[i] = ((p shr 16) and 0xFF) * norm            // R plane
            floats[area + i] = ((p shr 8) and 0xFF) * norm      // G plane
            floats[2 * area + i] = (p and 0xFF) * norm          // B plane
        }
        buffer.clear()
        buffer.asFloatBuffer().put(floats)
        return Tensor(buffer, DataType.Float32, intArrayOf(1, 3, h, w))
    }
}
