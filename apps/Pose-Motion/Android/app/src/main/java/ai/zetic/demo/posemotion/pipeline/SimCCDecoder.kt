package ai.zetic.demo.posemotion.pipeline

import ai.zetic.demo.posemotion.AppConfig
import android.graphics.RectF
import kotlin.math.exp
import kotlin.math.min

/**
 * Decodes RTMPose SimCC outputs: per joint, argmax over the x/y bin rows,
 * divided by the split ratio → crop-pixel coords → normalized frame coords.
 * Confidence = min of the two softmax peak probabilities (the outputs are logits).
 */
object SimCCDecoder {

    fun decode(
        simccX: FloatArray,   // [17 * 384]
        simccY: FloatArray,   // [17 * 512]
        cropPx: RectF,
        frameW: Float,
        frameH: Float,
    ): List<Keypoint2D> {
        val joints = AppConfig.JOINT_COUNT
        val xBins = simccX.size / joints
        val yBins = simccY.size / joints
        val split = AppConfig.SIMCC_SPLIT_RATIO
        val cropW = AppConfig.POSE_INPUT_WIDTH.toFloat()
        val cropH = AppConfig.POSE_INPUT_HEIGHT.toFloat()

        val result = ArrayList<Keypoint2D>(joints)
        for (j in 0 until joints) {
            val (xIdx, xConf) = peak(simccX, j * xBins, xBins)
            val (yIdx, yConf) = peak(simccY, j * yBins, yBins)

            val xCrop = xIdx / split          // 0..192 crop-pixel space
            val yCrop = yIdx / split          // 0..256
            val frameX = cropPx.left + xCrop / cropW * cropPx.width()
            val frameY = cropPx.top + yCrop / cropH * cropPx.height()

            result.add(Keypoint2D(frameX / frameW, frameY / frameH, min(xConf, yConf)))
        }
        return result
    }

    /** Returns (argmax index, softmax peak probability) for one bin row. */
    private fun peak(data: FloatArray, offset: Int, count: Int): Pair<Int, Float> {
        var maxValue = Float.NEGATIVE_INFINITY
        var maxIndex = 0
        for (i in 0 until count) {
            val v = data[offset + i]
            if (v > maxValue) {
                maxValue = v
                maxIndex = i
            }
        }
        // softmax peak = 1 / Σ exp(v - max)
        var sum = 0f
        for (i in 0 until count) sum += exp(data[offset + i] - maxValue)
        return maxIndex to if (sum > 0f) 1f / sum else 0f
    }
}
