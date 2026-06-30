package com.brew.llm

import android.content.Context
import com.brew.BrewConfig
import com.zeticai.mlange.core.model.llm.LLMModelMode
import com.zeticai.mlange.core.model.llm.ZeticMLangeLLMModel
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

/**
 * Single owner of the on-device Gemma model. All native access runs on one
 * dedicated thread (the model must be constructed on, and used from, the same
 * thread). Exposes a streaming, sanitized generation that throttles UI updates.
 */
class LLMService(private val context: Context) {

    private val exec = Executors.newSingleThreadExecutor { r -> Thread(r, "brew-llm") }
    private val dispatcher = exec.asCoroutineDispatcher()

    private var model: ZeticMLangeLLMModel? = null
    @Volatile private var cancelRequested = false

    val isLoaded: Boolean get() = model != null

    suspend fun ensureLoaded(onProgress: (Float) -> Unit = {}) = withContext(dispatcher) {
        if (model != null) return@withContext
        var lastError: Throwable? = null
        repeat(BrewConfig.MODEL_LOAD_ATTEMPTS) { attempt ->
            try {
                model = ZeticMLangeLLMModel(
                    context,
                    BrewConfig.PERSONAL_KEY,
                    BrewConfig.LLM_MODEL,
                    BrewConfig.LLM_VERSION,
                    LLMModelMode.RUN_AUTO,
                    onProgress = { p -> onProgress(p) },
                )
                return@withContext
            } catch (t: Throwable) {
                android.util.Log.w("BrewLLM", "LLM load attempt ${attempt + 1} failed", t)
                lastError = t
                // The SDK requires releasing a partial native handle before re-creating.
                try { model?.deinit() } catch (_: Throwable) {}
                model = null
                if (attempt < BrewConfig.MODEL_LOAD_ATTEMPTS - 1) delay(1_000L * (attempt + 1))
            }
        }
        throw lastError ?: IllegalStateException("LLM failed to load")
    }

    suspend fun unload() = withContext(dispatcher) {
        model?.deinit()
        model = null
    }

    fun cancel() {
        cancelRequested = true
    }

    /**
     * Streams a generation, applying [LLMOutput.sanitize] to the accumulated raw
     * text. [onUpdate] is invoked on the first token, then at most every 100 ms
     * (avoids O(n^2) sanitize churn), and once more with the final text.
     * Returns the final sanitized string.
     */
    suspend fun generateSanitized(
        prompt: String,
        maxTokens: Int = 1024,
        onUpdate: (String) -> Unit = {},
    ): String = withContext(dispatcher) {
        val llm = model ?: error("LLM not loaded")
        cancelRequested = false
        val raw = StringBuilder()
        var generated = 0
        var lastEmit = 0L
        var emittedAny = false

        llm.cleanUp()
        try {
            llm.run(prompt)
            while (true) {
                if (cancelRequested) break
                val result = llm.waitForNextToken()
                if (result.generatedTokens == 0) break
                if (result.token.isNotEmpty()) {
                    raw.append(result.token)
                    val now = System.currentTimeMillis()
                    if (!emittedAny || now - lastEmit >= 100) {
                        onUpdate(LLMOutput.sanitize(raw.toString()))
                        lastEmit = now
                        emittedAny = true
                    }
                }
                generated++
                if (generated >= maxTokens) break
            }
        } finally {
            llm.cleanUp()
        }
        val final = LLMOutput.sanitize(raw.toString())
        onUpdate(final)
        final
    }
}
