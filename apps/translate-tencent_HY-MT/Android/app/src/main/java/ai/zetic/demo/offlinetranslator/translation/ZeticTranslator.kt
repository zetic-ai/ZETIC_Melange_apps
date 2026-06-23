package ai.zetic.demo.offlinetranslator.translation

import android.content.Context
import ai.zetic.demo.offlinetranslator.config.ZeticConfig
import com.zeticai.mlange.core.model.llm.LLMModelMode
import com.zeticai.mlange.core.model.llm.ZeticMLangeLLMModel

/**
 * Real on-device translator backed by the ZETIC.ai Melange SDK (`com.zeticai.mlange:mlange`),
 * following the provided 1.6.1 deployment recipe. Mirrors iOS `ZeticTranslator`.
 *
 * Threading: every method here is invoked from the ViewModel's single dedicated engine thread.
 * The native model is created inside [load] (NOT in a field initializer) so the blocking native
 * init runs on that same thread — constructing it off-thread crashes native init.
 */
class ZeticTranslator(private val context: Context) : Translator {
    private var model: ZeticMLangeLLMModel? = null

    override fun load(onProgress: (Double) -> Unit) {
        // Release any previously loaded instance before re-init (e.g. a retry after a
        // failed load) so we never leak the old native model. The SDK requires deinit()
        // before re-creating a model.
        runCatching { model?.deinit() }
        model = null
        // Blocking native init; first run downloads the model (progress 0.0..1.0).
        model = ZeticMLangeLLMModel(
            context,
            ZeticConfig.personalKey,
            ZeticConfig.modelName,
            version = ZeticConfig.modelVersion,
            modelMode = LLMModelMode.RUN_AUTO,
            onProgress = { progress -> onProgress(progress.toDouble()) },
        )
    }

    override fun generate(prompt: String, onToken: (String) -> Boolean) {
        val model = this.model ?: throw TranslatorNotLoadedException()
        model.run(prompt)
        while (true) {
            val result = model.waitForNextToken() // blocks until the next token is ready
            if (result.generatedTokens == 0) break
            val token = result.token
            if (token.isNotEmpty()) {
                if (!onToken(token)) break // cooperative cancel between tokens
            }
        }
    }

    override fun reset() {
        // Reset KV-cache / conversation state from any prior turn, keep model loaded.
        runCatching { model?.cleanUp() }
    }

    override fun tearDown() {
        runCatching { model?.deinit() } // release native resources
        model = null
    }
}
