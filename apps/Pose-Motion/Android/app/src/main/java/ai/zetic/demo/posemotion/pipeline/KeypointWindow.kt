package ai.zetic.demo.posemotion.pipeline

import ai.zetic.demo.posemotion.AppConfig
import kotlin.math.min

/**
 * Sliding window of the last T frames of 2D keypoints, converted COCO-17 → H36M-17
 * and normalized for MotionBERT ([1,T,17,3] with x,y centered and divided by
 * min(W,H)/2). Confined to the inference thread.
 */
class KeypointWindow {
    private val t = AppConfig.LIFT_WINDOW
    private val frames = ArrayDeque<FloatArray>()   // each 17*3 floats, chronological

    val hasData: Boolean get() = frames.isNotEmpty()

    fun reset() = frames.clear()

    /** `coco` is 17 keypoints in COCO order, normalized 0..1 frame coordinates. */
    fun push(coco: List<Keypoint2D>, frameW: Float, frameH: Float) {
        if (coco.size != 17) return
        val h36m = cocoToH36M(coco)
        // MotionBERT image normalization: (px - W/2) / (min(W,H)/2), same for y.
        val halfMin = min(frameW, frameH) / 2f

        val frame = FloatArray(17 * 3)
        for (i in h36m.indices) {
            val kp = h36m[i]
            frame[i * 3] = (kp.x * frameW - frameW / 2f) / halfMin
            frame[i * 3 + 1] = (kp.y * frameH - frameH / 2f) / halfMin
            frame[i * 3 + 2] = kp.conf
        }
        frames.addLast(frame)
        while (frames.size > t) frames.removeFirst()
    }

    /** Flat [1,T,17,3] floats, left-padded by repeating the oldest frame; null if empty. */
    fun windowFloats(): FloatArray? {
        val first = frames.firstOrNull() ?: return null
        val out = FloatArray(t * 17 * 3)
        var offset = 0
        repeat(t - frames.size) {
            first.copyInto(out, offset); offset += 51
        }
        for (f in frames) {
            f.copyInto(out, offset); offset += 51
        }
        return out
    }

    companion object {
        /** Standard COCO-17 → H36M-17 reorder (midpoints average position and confidence). */
        fun cocoToH36M(c: List<Keypoint2D>): List<Keypoint2D> {
            fun mid(a: Keypoint2D, b: Keypoint2D) =
                Keypoint2D((a.x + b.x) / 2, (a.y + b.y) / 2, (a.conf + b.conf) / 2)
            val pelvis = mid(c[11], c[12])
            val thorax = mid(c[5], c[6])
            return listOf(
                pelvis,              // 0
                c[12],               // 1  R hip
                c[14],               // 2  R knee
                c[16],               // 3  R ankle
                c[11],               // 4  L hip
                c[13],               // 5  L knee
                c[15],               // 6  L ankle
                mid(pelvis, thorax), // 7  spine
                thorax,              // 8
                c[0],                // 9  neck (nose)
                mid(c[1], c[2]),     // 10 head (mid-eye)
                c[5],                // 11 L shoulder
                c[7],                // 12 L elbow
                c[9],                // 13 L wrist
                c[6],                // 14 R shoulder
                c[8],                // 15 R elbow
                c[10],               // 16 R wrist
            )
        }
    }
}
