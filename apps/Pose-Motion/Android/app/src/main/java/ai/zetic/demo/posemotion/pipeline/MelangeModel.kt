package ai.zetic.demo.posemotion.pipeline

import ai.zetic.demo.posemotion.AppConfig
import ai.zetic.demo.posemotion.benchmark.MemoryProbe
import android.content.Context
import android.util.Log
import com.zeticai.mlange.core.model.ModelMode
import com.zeticai.mlange.core.model.ZeticMLangeModel
import com.zeticai.mlange.core.tensor.Tensor

/**
 * Thin wrapper over ZeticMLangeModel: loads once, runs inference, times latency.
 * Native init is thread-affine — load() and run() must both happen on the
 * inference thread. deinit() is required before re-init (retry) and at teardown.
 */
class MelangeModel(val label: String) {
    private var model: ZeticMLangeModel? = null
    private var firstRunDone = false

    var lastLatencyMs: Double = 0.0
        private set

    val isLoaded: Boolean get() = model != null

    fun load(context: Context, name: String, version: Int, onDownload: (Float) -> Unit) {
        deinit()   // SDK requires deinit before re-creating (e.g. retry after failed load)
        model = ZeticMLangeModel(
            context.applicationContext,
            AppConfig.PERSONAL_KEY,
            name,
            version = version,
            modelMode = ModelMode.RUN_AUTO,   // measured faster than RUN_SPEED on Exynos 2400
            onDownload = onDownload,
        )
        MemoryProbe.log("$label loaded")
    }

    fun deinit() {
        // core 0.1.1 (mlange 1.8.1) exposes Closeable.close() (deinit() was pre-0.1 API).
        runCatching { model?.close() }
        model = null
        firstRunDone = false
    }

    fun run(inputs: Array<Tensor>): Array<Tensor> {
        val m = model ?: throw IllegalStateException("$label not loaded")
        if (!firstRunDone) Log.d("run", "[run] $label first inference…")
        val t0 = System.nanoTime()
        val outputs = m.run(inputs) ?: throw IllegalStateException("$label returned null")
        lastLatencyMs = (System.nanoTime() - t0) / 1_000_000.0
        if (!firstRunDone) {
            firstRunDone = true
            Log.d("run", "[run] $label ok (%.1f ms)".format(lastLatencyMs))
        }
        return outputs
    }

    companion object {
        /** Output tensor payload as floats (shape is private; callers know the sizes). */
        fun floats(tensor: Tensor): FloatArray {
            val buf = tensor.data
            buf.rewind()
            val out = FloatArray(buf.remaining() / 4)
            buf.asFloatBuffer().get(out)
            return out
        }
    }
}
