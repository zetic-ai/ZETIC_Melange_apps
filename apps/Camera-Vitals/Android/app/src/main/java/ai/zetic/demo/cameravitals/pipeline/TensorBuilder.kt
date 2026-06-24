package ai.zetic.demo.cameravitals.pipeline

import ai.zetic.demo.cameravitals.AppConfig
import com.zeticai.mlange.core.tensor.DataType
import com.zeticai.mlange.core.tensor.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * Builds the model input Tensor from a flattened 31-frame clip, applying the rPPG-Toolbox
 * 'Standardized' preprocessing: single scalar (x - mean) / std, written straight into a
 * direct native-order ByteBuffer.
 */
object TensorBuilder {
    fun build(flat: FloatArray): Tensor? {
        val n = flat.size

        var mean = 0f
        for (v in flat) mean += v
        mean /= n

        var sumsq = 0f
        for (v in flat) { val d = v - mean; sumsq += d * d }
        val std = sqrt(sumsq / n)
        if (std <= 1e-6f) return null   // flat clip → skip (avoid NaNs)
        val invStd = 1f / std

        val buffer = ByteBuffer.allocateDirect(n * 4).order(ByteOrder.nativeOrder())
        val fb = buffer.asFloatBuffer()
        for (v in flat) fb.put((v - mean) * invStd)
        buffer.rewind()

        return Tensor(
            buffer,
            DataType.Float32,
            intArrayOf(AppConfig.FRAMES_IN, AppConfig.CHANNELS, AppConfig.IMG_SIZE, AppConfig.IMG_SIZE)
        )
    }
}
