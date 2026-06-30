package com.brew.engine

import android.content.Context
import com.brew.asr.WhisperService
import com.brew.llm.LLMService
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/** Download/prepare phase of the on-device AI model (mirrors iOS `Phase`). */
sealed interface ModelPhase {
    data object Idle : ModelPhase
    data class Downloading(val progress: Float) : ModelPhase
    data object Preparing : ModelPhase
    data object Ready : ModelPhase
    data class Failed(val message: String) : ModelPhase
}

/**
 * Process-wide coordinator enforcing that the Gemma LLM and the Whisper
 * encoder/decoder are **never co-resident** (they would OOM mid-range devices).
 * A single [Mutex] serializes all native work, so a transcription and a
 * generation can never run at the same time. Acquiring one family unloads the
 * other.
 */
class ModelCoordinator(context: Context) {
    val whisper = WhisperService(context)
    val llm = LLMService(context)

    private val lock = Mutex()

    private val _llmPhase = MutableStateFlow<ModelPhase>(ModelPhase.Idle)
    val llmPhase: StateFlow<ModelPhase> = _llmPhase.asStateFlow()

    /** Runs [block] with the Gemma LLM loaded (unloading Whisper first). */
    suspend fun <T> withLlm(block: suspend (LLMService) -> T): T = lock.withLock {
        if (whisper.isLoaded) whisper.unload()
        try {
            if (!llm.isLoaded) {
                _llmPhase.value = ModelPhase.Preparing
                llm.ensureLoaded { p -> _llmPhase.value = phaseFor(p) }
            }
            _llmPhase.value = ModelPhase.Ready
        } catch (t: Throwable) {
            android.util.Log.e("BrewLLM", "Gemma load failed", t)
            _llmPhase.value = ModelPhase.Failed(t.message ?: "Model unavailable")
            throw t
        }
        block(llm)
    }

    /**
     * Transcribes [audio16kMono] with Whisper (unloading the LLM first), then
     * frees the Whisper models so a subsequent enhance can load Gemma.
     */
    suspend fun transcribe(audio16kMono: FloatArray, onProgress: (Float) -> Unit = {}): String =
        lock.withLock {
            if (llm.isLoaded) llm.unload()
            try {
                whisper.ensureLoaded()
                whisper.transcribe(audio16kMono, onProgress)
            } finally {
                whisper.unload()
            }
        }

    /** Pre-warms the LLM download so the model-status chip reflects readiness. */
    suspend fun preloadLlm() {
        try {
            withLlm { }
        } catch (_: Throwable) {
            // Phase already set to Failed; recording/transcription stay usable.
        }
    }

    fun retryLlm() {
        _llmPhase.value = ModelPhase.Idle
    }

    private fun phaseFor(p: Float): ModelPhase =
        if (p > 0f && p < 1f) ModelPhase.Downloading(p) else ModelPhase.Preparing
}
