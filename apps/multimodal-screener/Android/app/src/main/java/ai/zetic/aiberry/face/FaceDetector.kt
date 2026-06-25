package ai.zetic.aiberry.face

import android.graphics.Bitmap
import android.graphics.Rect
import ai.zetic.aiberry.core.AppConfig
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlin.math.max
import kotlin.math.roundToInt

/** A square 224x224 face crop plus the detector's confidence (0..1). */
class CroppedFace(val bitmap: Bitmap, val confidence: Float)

/**
 * On-device face detection + square crop via Google ML Kit — the Android parallel to the
 * iOS app's Apple Vision `VNDetectFaceRectangles`. Picks the largest face, expands the box by
 * [AppConfig.Face.CROP_MARGIN], squares + clamps it, and scales to the model's 224x224 input.
 *
 * [detect] is synchronous (blocks on the ML Kit task) and is meant to be called from a
 * background thread — never the main thread.
 */
class FaceDetector {
    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .build(),
    )

    /** [upright] must already be rotated to a natural upright orientation. */
    fun detect(upright: Bitmap): CroppedFace? {
        val faces = try {
            Tasks.await(detector.process(InputImage.fromBitmap(upright, 0)))
        } catch (_: Throwable) {
            return null
        }
        if (faces.isEmpty()) return null

        // Largest face by bounding-box area.
        val face = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() } ?: return null
        val box = face.boundingBox

        val size = AppConfig.Face.INPUT_SIZE
        val crop = squareWithMargin(box, upright.width, upright.height) ?: return null
        val cropped = Bitmap.createBitmap(upright, crop.left, crop.top, crop.width(), crop.height())
        val scaled = if (cropped.width == size && cropped.height == size) {
            cropped
        } else {
            Bitmap.createScaledBitmap(cropped, size, size, true)
        }
        // ML Kit gives no per-detection confidence in FAST mode; treat a clean detection as 1.0
        // (clamped to the iOS minimum weight of 0.2 downstream).
        return CroppedFace(scaled.copy(Bitmap.Config.ARGB_8888, false), 1.0f)
    }

    /** Expand by margin, square (centered on the box center), clamp to image bounds. */
    private fun squareWithMargin(box: Rect, w: Int, h: Int): Rect? {
        val cx = box.exactCenterX()
        val cy = box.exactCenterY()
        val base = max(box.width(), box.height()).toFloat()
        val side = base * (1f + 2f * AppConfig.Face.CROP_MARGIN)
        var half = side / 2f
        // Don't request a region larger than the image.
        half = half.coerceAtMost(min(w, h) / 2f)
        var left = (cx - half).roundToInt()
        var top = (cy - half).roundToInt()
        var s = (half * 2f).roundToInt()
        if (s <= 0) return null
        // Clamp origin so the square stays inside the image.
        left = left.coerceIn(0, max(0, w - s))
        top = top.coerceIn(0, max(0, h - s))
        s = s.coerceAtMost(min(w - left, h - top))
        if (s <= 0) return null
        return Rect(left, top, left + s, top + s)
    }

    private fun min(a: Int, b: Int) = if (a < b) a else b
}
