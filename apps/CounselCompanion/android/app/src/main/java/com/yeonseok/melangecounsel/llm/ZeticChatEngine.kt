package com.yeonseok.melangecounsel.llm

import android.content.Context
import com.zeticai.mlange.core.model.llm.ZeticMLangeLLMModel
import com.zeticai.mlange.core.model.ModelLoadingStatus
import com.yeonseok.melangecounsel.domain.GenerationRecord
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.isActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

class ZeticChatEngine(private val context: Context) {
    private var model: ZeticMLangeLLMModel? = null
    private val mutex = Mutex()
    private val stopRequested = AtomicBoolean(false)

    val isInitialized: Boolean get() = model != null

    fun initialize(onLoadingStatus: (ModelLoadingStatus) -> Unit) {
        if (model == null) {
            model = ZeticMLangeLLMModel(context.applicationContext, PERSONAL_KEY, MODEL_ID) { status ->
                onLoadingStatus(status)
            }
        }
    }

    suspend fun generate(prompt: String, onToken: suspend (String) -> Unit): GenerationRecord {
        val llm = model ?: return GenerationRecord(
            fullText = "", tokenCount = 0, durationMs = 0,
            stopped = false, errorMessage = "Model not initialized"
        )
        return withContext(Dispatchers.IO) {
            mutex.withLock {
                stopRequested.set(false)
                llm.cleanUp()

                val start = System.currentTimeMillis()
                val completeText = StringBuilder()
                var tokenCount = 0
                var errorMessage: String? = null

                try {
                    llm.run(prompt)
                    coroutineScope {
                        while (isActive && !stopRequested.get()) {
                            ensureActive()
                            val tokenResult = llm.waitForNextToken()
                            if (tokenResult.generatedTokens == 0) break
                            if (tokenResult.token.isNotEmpty()) {
                                completeText.append(tokenResult.token)
                                tokenCount += tokenResult.generatedTokens
                                onToken(tokenResult.token)
                            }
                        }
                    }
                } catch (t: Throwable) {
                    errorMessage = t.message ?: "Generation failed"
                } finally {
                    llm.cleanUp()
                }

                val end = System.currentTimeMillis()
                GenerationRecord(
                    fullText = completeText.toString(),
                    tokenCount = tokenCount,
                    durationMs = end - start,
                    stopped = stopRequested.get(),
                    errorMessage = errorMessage
                )
            }
        }
    }

    fun requestStop() {
        stopRequested.set(true)
        model?.cleanUp()
    }

    fun destroy() {
        stopRequested.set(true)
        model?.cleanUp()
    }

    companion object {
        const val MODEL_ID: String = "Steve/kanana_1_5__2_1b_instruct"
        private const val PERSONAL_KEY: String = "YOUR_MLANGE_KEY"

        fun maskedPersonalKey(): String {
            return if (PERSONAL_KEY.length < 8) {
                "****"
            } else {
                PERSONAL_KEY.take(4) + "****" + PERSONAL_KEY.takeLast(4)
            }
        }
    }
}
