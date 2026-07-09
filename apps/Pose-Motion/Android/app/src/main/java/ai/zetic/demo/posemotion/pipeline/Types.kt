package ai.zetic.demo.posemotion.pipeline

import android.graphics.Bitmap
import android.graphics.RectF

/** Detection in normalized 0..1 frame coordinates. */
data class Detection(val rect: RectF, val score: Float, val classIndex: Int)

/** Keypoint in normalized 0..1 frame coordinates. */
data class Keypoint2D(val x: Float, val y: Float, val conf: Float)

data class Vec3(val x: Float, val y: Float, val z: Float)

data class FrameTimings(
    var detectMs: Double = 0.0,
    var poseMs: Double = 0.0,
    var liftMs: Double = 0.0,
    var totalMs: Double = 0.0,
)

data class BenchmarkSnapshot(
    val detectMs: Double = 0.0,
    val poseMs: Double = 0.0,
    val liftMs: Double = 0.0,
    val totalMs: Double = 0.0,
    val fps: Double = 0.0,
    val memoryMB: Double = 0.0,
    val peakMemoryMB: Double = 0.0,
)

/** One processed frame: the display bitmap and every overlay travels together (frame-synced). */
data class FrameResult(
    val bitmap: Bitmap,
    val frameWidth: Int,
    val frameHeight: Int,
    val personBox: RectF?,          // normalized
    val ballBox: RectF?,            // normalized
    val keypoints: List<Keypoint2D>?,   // COCO-17, normalized frame coords
    val pose3D: List<Vec3>?,        // H36M-17, root-relative
    val timings: FrameTimings,
)
