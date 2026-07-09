package ai.zetic.demo.cherrypad.llm

import android.content.Context
import android.os.Handler
import android.os.Looper
import com.zeticai.mlange.core.model.llm.ZeticMLangeLLMModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

/**
 * Process-wide owner of the on-device LFM2.5-350M model, shared by the container app
 * and the CherryPad keyboard (they run in the same process). All native access runs on
 * one dedicated thread — the model must be constructed on, and used from, the same
 * thread. Loaded once, kept WARM the whole session; only `cleanUp()` (KV-cache reset)
 * runs between requests.
 */
object LLMService {

    sealed interface Phase {
        data object Idle : Phase
        data class Downloading(val progress: Float) : Phase
        data object Preparing : Phase
        data object Ready : Phase
        data class Failed(val message: String) : Phase
    }

    private val exec = Executors.newSingleThreadExecutor { r -> Thread(r, "cherrypad-llm") }
    private val dispatcher = exec.asCoroutineDispatcher()
    // Callers assign Compose state from callbacks, so all callbacks are delivered on main.
    private val mainHandler = Handler(Looper.getMainLooper())
    private val loadMutex = Mutex()

    private var model: ZeticMLangeLLMModel? = null
    @Volatile private var cancelRequested = false

    private val _phase = MutableStateFlow<Phase>(Phase.Idle)
    val phase: StateFlow<Phase> = _phase.asStateFlow()

    val isReady: Boolean get() = model != null && _phase.value is Phase.Ready
    val loadError: String? get() = (_phase.value as? Phase.Failed)?.message

    /**
     * Idempotent load. Concurrent callers await the same in-flight load via [loadMutex].
     * Emits [Phase.Downloading] while the model file downloads, [Phase.Preparing] while it
     * initializes, then [Phase.Ready]. On failure releases the partial native handle so a
     * retry can construct cleanly. [onProgress] is delivered on the main thread.
     */
    suspend fun ensureLoaded(context: Context, onProgress: (Float) -> Unit = {}) {
        if (model != null) {
            _phase.value = Phase.Ready
            return
        }
        val appContext = context.applicationContext
        loadMutex.withLock {
            if (model != null) {
                _phase.value = Phase.Ready
                return
            }
            withContext(dispatcher) {
                _phase.value = Phase.Preparing
                try {
                    val m = ZeticMLangeLLMModel(
                        appContext,
                        ZeticConfig.PERSONAL_KEY,
                        ZeticConfig.MODEL_NAME,
                        ZeticConfig.MODEL_VERSION,
                        ZeticConfig.MODEL_MODE,
                        onProgress = { p ->
                            _phase.value = if (p > 0f && p < 1f) Phase.Downloading(p) else Phase.Preparing
                            mainHandler.post { onProgress(p) }
                        },
                    )
                    model = m
                    _phase.value = Phase.Ready
                } catch (t: Throwable) {
                    android.util.Log.w("CherryPadLLM", "LLM load failed", t)
                    // The SDK requires releasing a partial native handle before re-creating.
                    try { model?.deinit() } catch (_: Throwable) {}
                    model = null
                    _phase.value = Phase.Failed(t.message ?: "The on-device model isn't ready yet.")
                    throw t
                }
            }
        }
    }

    fun cancel() {
        cancelRequested = true
    }

    suspend fun unload() = withContext(dispatcher) {
        try { model?.deinit() } catch (_: Throwable) {}
        model = null
        _phase.value = Phase.Idle
    }

    /**
     * Streams a generation, applying [LLMOutput.sanitize] to the accumulated raw text.
     * [onUpdate] is invoked on the first token, then at most every 100 ms (avoids O(n^2)
     * sanitize churn), and once more with the final text, always on the main thread.
     * Returns the final sanitized string.
     */
    suspend fun generateSanitized(
        prompt: String,
        maxTokens: Int,
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
                        val partial = LLMOutput.sanitize(raw.toString())
                        withContext(Dispatchers.Main) { onUpdate(partial) }
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
        withContext(Dispatchers.Main) { onUpdate(final) }
        final
    }
}
