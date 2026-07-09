package ai.zetic.demo.posemotion.pipeline

import ai.zetic.demo.posemotion.AppConfig
import ai.zetic.demo.posemotion.benchmark.BenchmarkStats
import ai.zetic.demo.posemotion.video.DecodedFrame
import android.graphics.RectF
import com.zeticai.mlange.core.tensor.DataType
import com.zeticai.mlange.core.tensor.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The three-model chain: YOLO26n (person + ball) → RTMPose (crop → 2D skeleton)
 * → MotionBERT-Lite (keypoint window → 3D). Everything runs on the inference thread.
 * Line-for-line port of the iOS PosePipeline.
 */
class PosePipeline {
    val detector = MelangeModel("YOLO26n")
    val pose = MelangeModel("RTMPose-s")
    val lift = MelangeModel("MotionBERT-Lite")

    /** Called on the inference thread with each processed frame. */
    var onResult: ((FrameResult, BenchmarkSnapshot) -> Unit)? = null

    private val preprocessor = FramePreprocessor()
    private val window = KeypointWindow()
    private val stats = BenchmarkStats()
    private var frameCounter = 0
    private var lastKeypoints: List<Keypoint2D>? = null
    private var lastPose3D: List<Vec3>? = null

    private val liftBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(AppConfig.LIFT_WINDOW * 17 * 3 * 4)
        .order(ByteOrder.nativeOrder())

    fun resetSession() {
        window.reset()
        frameCounter = 0
        lastKeypoints = null
        lastPose3D = null
        stats.resetFps()
    }

    /** One-time first inference per model on zero inputs, so the rolling stats
     *  never contain the backend compile spike. Inference thread only. */
    fun warmup() {
        if (detector.isLoaded) {
            val zeros = ByteBuffer.allocateDirect(3 * 640 * 640 * 4).order(ByteOrder.nativeOrder())
            runCatching { detector.run(arrayOf(Tensor(zeros, DataType.Float32, intArrayOf(1, 3, 640, 640)))) }
        }
        if (pose.isLoaded) {
            val zeros = ByteBuffer.allocateDirect(3 * 256 * 192 * 4).order(ByteOrder.nativeOrder())
            runCatching { pose.run(arrayOf(Tensor(zeros, DataType.Float32, intArrayOf(1, 3, 256, 192)))) }
        }
        if (lift.isLoaded) {
            val zeros = ByteBuffer.allocateDirect(81 * 17 * 3 * 4).order(ByteOrder.nativeOrder())
            runCatching { lift.run(arrayOf(Tensor(zeros, DataType.Float32, intArrayOf(1, 81, 17, 3)))) }
        }
    }

    fun process(frame: DecodedFrame) {
        val t0 = System.nanoTime()
        val timings = FrameTimings()
        val bitmap = frame.bitmap
        val frameW = bitmap.width.toFloat()
        val frameH = bitmap.height.toFloat()

        // 1. Detector: person + ball
        var person: Detection? = null
        var ball: Detection? = null
        if (detector.isLoaded) {
            runCatching {
                val outputs = detector.run(arrayOf(preprocessor.detectorTensor(bitmap)))
                val data = MelangeModel.floats(outputs[0])
                val decoded = DetectorDecoder.decode(data)
                person = decoded.first
                ball = decoded.second
                timings.detectMs = detector.lastLatencyMs
            }
        }

        // 2. Pose: crop → SimCC → keypoints
        var keypoints: List<Keypoint2D>? = null
        val personBox = person?.rect
        if (pose.isLoaded && personBox != null) {
            val crop = PersonCropper.cropRect(personBox, frameW, frameH)
            if (crop.width() > 4 && crop.height() > 4) {
                runCatching {
                    val outputs = pose.run(arrayOf(preprocessor.poseTensor(bitmap, crop)))
                    if (outputs.size >= 2) {
                        // Identify simcc_x vs simcc_y by size (Tensor shape is private):
                        // x has 17*384 = 6528 floats, y has 17*512 = 8704.
                        val a = MelangeModel.floats(outputs[0])
                        val b = MelangeModel.floats(outputs[1])
                        val (sx, sy) = if (a.size <= b.size) a to b else b to a
                        keypoints = SimCCDecoder.decode(sx, sy, crop, frameW, frameH)
                        timings.poseMs = pose.lastLatencyMs
                    }
                }
            }
        }

        // 3. Keypoint window (repeat last frame on a miss so the lift window stays coherent)
        val kps = keypoints
        if (kps != null) {
            window.push(kps, frameW, frameH)
            lastKeypoints = kps
        } else {
            lastKeypoints?.let { last ->
                window.push(last.map { Keypoint2D(it.x, it.y, 0.1f) }, frameW, frameH)
            }
        }

        // 4. 3D lift, every LIFT_STRIDE frames
        if (lift.isLoaded && frameCounter % AppConfig.LIFT_STRIDE == 0 && window.hasData) {
            window.windowFloats()?.let { floats ->
                runCatching {
                    liftBuffer.clear()
                    liftBuffer.asFloatBuffer().put(floats)
                    val input = Tensor(
                        liftBuffer, DataType.Float32,
                        intArrayOf(1, AppConfig.LIFT_WINDOW, 17, 3)
                    )
                    val outputs = lift.run(arrayOf(input))
                    readPose3D(MelangeModel.floats(outputs[0]))?.let { lastPose3D = it }
                    timings.liftMs = lift.lastLatencyMs
                }
            }
        }
        frameCounter++

        timings.totalMs = (System.nanoTime() - t0) / 1_000_000.0
        stats.record(timings)
        if (frameCounter % 30 == 0) {
            val s = stats.snapshot()
            android.util.Log.d(
                "stats",
                "[stats] fps=%.1f det=%.1f pose=%.1f lift=%.1f total=%.1f mem=%.0f"
                    .format(s.fps, s.detectMs, s.poseMs, s.liftMs, s.totalMs, s.memoryMB)
            )
        }

        val result = FrameResult(
            bitmap = bitmap,
            frameWidth = bitmap.width,
            frameHeight = bitmap.height,
            personBox = personBox,
            ballBox = ball?.rect,
            keypoints = keypoints,
            pose3D = lastPose3D,
            timings = timings,
        )
        onResult?.invoke(result, stats.snapshot())
    }

    fun deinitAll() {
        detector.deinit()
        pose.deinit()
        lift.deinit()
    }

    /** [1,T,17,3] flat floats → the configured window frame as 17 root-relative joints. */
    private fun readPose3D(floats: FloatArray): List<Vec3>? {
        if (floats.size % 51 != 0) return null
        val frames = floats.size / 51
        if (frames == 0) return null
        val t = minOf(AppConfig.LIFT_READ_INDEX, frames - 1)
        val base = t * 51
        return (0 until 17).map { j ->
            Vec3(floats[base + j * 3], floats[base + j * 3 + 1], floats[base + j * 3 + 2])
        }
    }
}
