package ai.zetic.demo.cameravitals.vision

import android.graphics.Bitmap
import android.graphics.Rect
import ai.zetic.demo.cameravitals.util.clamped
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlin.math.max
import kotlin.math.min

/**
 * Detects the largest face with ML Kit and returns a stable, enlarged, squared ROI in the
 * bitmap's pixel coordinates. Smooths the box and holds the last good box briefly when detection
 * drops, so cropping stays steady. (Android analogue of the iOS Vision FaceROITracker.)
 */
class FaceRoiTracker {
    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .build()
    )

    private var smoothed: Rect? = null
    private var missCount = 0
    private val maxMiss = 8
    private val smoothing = 0.35f
    private val enlarge = 1.5f

    /** Runs detection synchronously on the calling (background) thread. */
    fun detect(bitmap: Bitmap): Rect? {
        val image = InputImage.fromBitmap(bitmap, 0)
        val faces: List<Face> = try {
            Tasks.await(detector.process(image))
        } catch (e: Exception) {
            return hold()
        }
        val face = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() } ?: return hold()

        val box = squareEnlarged(face.boundingBox, bitmap.width, bitmap.height)
        missCount = 0
        val s = smoothed
        smoothed = if (s != null) {
            Rect(
                (s.left + smoothing * (box.left - s.left)).toInt(),
                (s.top + smoothing * (box.top - s.top)).toInt(),
                (s.right + smoothing * (box.right - s.right)).toInt(),
                (s.bottom + smoothing * (box.bottom - s.bottom)).toInt()
            )
        } else {
            box
        }
        return smoothed
    }

    fun reset() {
        smoothed = null
        missCount = 0
    }

    private fun hold(): Rect? {
        missCount++
        if (missCount > maxMiss) { smoothed = null; return null }
        return smoothed
    }

    private fun squareEnlarged(r: Rect, w: Int, h: Int): Rect {
        val side = (max(r.width(), r.height()) * enlarge).toInt().clamped(1, min(w, h))
        val cx = r.centerX()
        val cy = r.centerY()
        val x = (cx - side / 2).clamped(0, max(0, w - side))
        val y = (cy - side / 2).clamped(0, max(0, h - side))
        return Rect(x, y, x + side, y + side)
    }
}
