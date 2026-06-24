package ai.zetic.demo.cameravitals.pipeline

import android.content.Context
import ai.zetic.demo.cameravitals.AppConfig
import com.zeticai.mlange.core.model.ModelMode
import com.zeticai.mlange.core.model.ZeticMLangeModel
import com.zeticai.mlange.core.tensor.Tensor

/** Thin wrapper over ZeticMLangeModel: loads once, runs inference, times latency. */
class RppgModel {
    private var model: ZeticMLangeModel? = null

    @Volatile
    var lastLatencyMs: Double = 0.0
        private set

    /** Loads/downloads the model on a background thread. Callbacks fire on that thread. */
    fun load(context: Context, onProgress: (Float) -> Unit, onResult: (Throwable?) -> Unit) {
        Thread {
            try {
                model = ZeticMLangeModel(
                    context.applicationContext,
                    AppConfig.PERSONAL_KEY,
                    AppConfig.MODEL_NAME,
                    version = AppConfig.MODEL_VERSION,
                    modelMode = ModelMode.RUN_AUTO,
                    onDownload = { progress -> onProgress(progress) }   // 1.8.1 names this `onDownload`
                )
                onResult(null)
            } catch (t: Throwable) {
                onResult(t)
            }
        }.start()
    }

    /** Runs inference and returns the 30-sample rPPG waveform. */
    fun infer(input: Tensor): FloatArray {
        val m = model ?: throw IllegalStateException("Model not loaded")
        val t0 = System.nanoTime()
        val outputs = m.run(arrayOf(input)) ?: throw IllegalStateException("Inference returned null")
        lastLatencyMs = (System.nanoTime() - t0) / 1_000_000.0
        val buf = outputs[0].data
        buf.rewind()
        val count = buf.remaining() / 4
        val out = FloatArray(count)
        buf.asFloatBuffer().get(out)
        return out
    }
}
