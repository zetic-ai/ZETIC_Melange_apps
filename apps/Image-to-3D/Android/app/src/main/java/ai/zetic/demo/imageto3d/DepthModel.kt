package ai.zetic.demo.imageto3d

import android.content.Context
import android.util.Log
import com.zeticai.mlange.core.model.ModelMode
import com.zeticai.mlange.core.model.QuantType
import com.zeticai.mlange.core.model.ZeticMLangeModel
import com.zeticai.mlange.core.tensor.DataType
import com.zeticai.mlange.core.tensor.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Depth map decoded from the model's output tensor (relative inverse depth,
 *  larger = closer), with a robust 2/98-percentile range. */
class DepthMap(val values: FloatArray, val width: Int, val height: Int) {
    val robustLo: Float
    val robustHi: Float

    init {
        val sample = FloatArray((values.size + 15) / 16)
        var j = 0
        var i = 0
        while (i < values.size) {
            sample[j++] = values[i]
            i += 16
        }
        sample.sort()
        val lo = sample[((sample.size - 1) * 0.02f).toInt()]
        val hi = sample[((sample.size - 1) * 0.98f).toInt()]
        robustLo = lo
        robustHi = maxOf(hi, lo + 1e-6f)
    }

    /** Disparity normalized to [0, 1] over the robust range; 1 = closest. */
    fun normalized(index: Int): Float =
        ((values[index] - robustLo) / (robustHi - robustLo)).coerceIn(0f, 1f)
}

/** Thin wrapper over ZeticMLangeModel: loads once, runs inference, times latency. */
class DepthModel {
    private var model: ZeticMLangeModel? = null
    var lastInferMs = 0.0
        private set

    /** Blocking; call from a background thread. */
    fun load(context: Context, onProgress: (Float) -> Unit) {
        model = ZeticMLangeModel(
            context,
            AppConfig.PERSONAL_KEY,
            AppConfig.MODEL_NAME,
            null,                 // version: latest
            ModelMode.RUN_AUTO,
            QuantType.FP32,
            { p: Float -> onProgress(p) },
        )
        Log.i(TAG, "model loaded: ${AppConfig.MODEL_NAME}")
    }

    fun infer(chw: FloatArray): DepthMap {
        val model = model ?: error("model not loaded")
        val size = AppConfig.INPUT_SIZE

        val buffer = ByteBuffer.allocateDirect(chw.size * 4).order(ByteOrder.nativeOrder())
        buffer.asFloatBuffer().put(chw)
        buffer.rewind()
        val input = Tensor(buffer, DataType.Float32, intArrayOf(1, 3, size, size))

        val t0 = System.nanoTime()
        val outputs = model.run(arrayOf(input))
        lastInferMs = (System.nanoTime() - t0) / 1e6

        val out = outputs.firstOrNull() ?: error("model returned no outputs")
        val bb = out.data.order(ByteOrder.nativeOrder())
        bb.rewind()
        val floats = FloatArray(bb.remaining() / 4)
        bb.asFloatBuffer().get(floats)

        // Defensive shape handling: expect H*W floats; sqrt fallback otherwise.
        val expected = size * size
        val (w, h) = if (floats.size == expected) {
            size to size
        } else {
            val s = Math.sqrt(floats.size.toDouble()).toInt()
            check(s * s == floats.size) { "unexpected output count ${floats.size}" }
            s to s
        }
        return DepthMap(floats, w, h)
    }

    fun deinit() {
        model?.close()
        model = null
    }

    private companion object { const val TAG = "ImageTo3D" }
}
