package ai.zetic.skinclassifier.classifier

import android.content.Context
import android.graphics.Bitmap
import ai.zetic.skinclassifier.core.AppConfig
import ai.zetic.skinclassifier.core.MelangeKit
import ai.zetic.skinclassifier.core.MelangeRuntime
import ai.zetic.skinclassifier.core.measureMs
import ai.zetic.skinclassifier.model.Classification
import ai.zetic.skinclassifier.model.SkinClass
import ai.zetic.skinclassifier.vision.ImagePreprocessor
import com.zeticai.mlange.core.model.ZeticMLangeModel
import java.util.concurrent.Callable

/**
 * Loads and runs the on-device skin classifier. All SDK calls are funnelled onto the shared
 * [MelangeRuntime.executor] (the SDK forbids concurrent init/run). 1:1 with the iOS
 * `SkinClassifier` service.
 */
class SkinClassifier(private val context: Context) {

    @Volatile
    private var model: ZeticMLangeModel? = null

    /** Idempotent: download + compile the model on first call, reporting 0..1 progress. */
    fun ensureLoaded(onProgress: (Float) -> Unit) {
        MelangeRuntime.executor.submit(Callable {
            ensureModel(onProgress)
        }).get()
    }

    /** Run the classifier on [bitmap], returning the ranked distribution + latency. */
    fun classify(bitmap: Bitmap): Classification {
        return MelangeRuntime.executor.submit(Callable {
            val m = ensureModel { }
            val values = ImagePreprocessor.toInput(bitmap)
            val input = MelangeKit.floatTensor(values, ImagePreprocessor.shape)
            val (outputs, ms) = measureMs { m.run(arrayOf(input)) }
            val first = outputs.firstOrNull() ?: error("Model returned no output")
            val logits = MelangeKit.floats(first)
            val expected = SkinClass.ordered.size
            if (logits.size < expected) error("Unexpected model output size ${logits.size}")
            Classification.fromLogits(logits, ms)
        }).get()
    }

    /** Release native resources; required before a fresh load on retry. */
    fun close() {
        model?.let {
            try {
                it.close()
            } catch (_: Throwable) {
            }
        }
        model = null
    }

    @Synchronized
    private fun ensureModel(onProgress: (Float) -> Unit): ZeticMLangeModel {
        model?.let { return it }
        val loaded = MelangeKit.load(
            context,
            AppConfig.Model.CLASSIFIER,
            AppConfig.Model.VERSION,
            onProgress,
        )
        model = loaded
        return loaded
    }
}
