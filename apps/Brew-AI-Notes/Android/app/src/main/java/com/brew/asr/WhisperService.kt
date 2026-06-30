package com.brew.asr

import android.content.Context
import com.brew.BrewConfig
import com.zeticai.mlange.core.model.ModelMode
import com.zeticai.mlange.core.model.ZeticMLangeModel
import com.zeticai.mlange.core.tensor.DataType
import com.zeticai.mlange.core.tensor.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.Future

/**
 * On-device speech-to-text via the ZeticMLange Whisper-small encoder/decoder
 * split. Owns both native models on a single dedicated thread (the SDK binds a
 * model handle to the thread that created it, and `run()` is synchronous).
 *
 * Pipeline per 30 s window: PCM -> log-mel [1,80,3000] -> encoder -> hidden
 * [1,1500,768] -> greedy decoder loop (forced English) -> detokenize.
 *
 * English is forced by prefilling the decoder prompt
 * `<|startoftranscript|><|en|><|transcribe|><|notimestamps|>` instead of letting
 * the model auto-detect the language.
 */
class WhisperService(private val context: Context) {

    private val exec = Executors.newSingleThreadExecutor { r -> Thread(r, "brew-whisper") }

    private var encoder: ZeticMLangeModel? = null
    private var decoder: ZeticMLangeModel? = null
    private var mel: LogMel? = null
    private var detok: WhisperDetokenizer? = null

    @Volatile var isLoaded = false
        private set

    /** Constructs both models + the mel/detokenizer on the engine thread (with retries). */
    fun ensureLoaded(onProgress: (Float) -> Unit = {}) {
        submit {
            if (isLoaded) return@submit
            mel = LogMel.fromAsset(context)
            detok = WhisperDetokenizer.fromAsset(context)
            var lastError: Throwable? = null
            repeat(BrewConfig.MODEL_LOAD_ATTEMPTS) { attempt ->
                try {
                    encoder = ZeticMLangeModel(
                        context,
                        BrewConfig.PERSONAL_KEY,
                        BrewConfig.WHISPER_ENCODER_MODEL,
                        BrewConfig.WHISPER_ENCODER_VERSION,
                        ModelMode.RUN_AUTO,
                        onProgress = { p -> onProgress(p * 0.5f) },
                    )
                    decoder = ZeticMLangeModel(
                        context,
                        BrewConfig.PERSONAL_KEY,
                        BrewConfig.WHISPER_DECODER_MODEL,
                        BrewConfig.WHISPER_DECODER_VERSION,
                        ModelMode.RUN_AUTO,
                        onProgress = { p -> onProgress(0.5f + p * 0.5f) },
                    )
                    isLoaded = true
                    return@submit
                } catch (t: Throwable) {
                    android.util.Log.w("BrewWhisper", "Whisper load attempt ${attempt + 1} failed", t)
                    lastError = t
                    // Release any partial handle (e.g. encoder loaded, decoder failed) before retry.
                    try { encoder?.close() } catch (_: Throwable) {}
                    try { decoder?.close() } catch (_: Throwable) {}
                    encoder = null
                    decoder = null
                    if (attempt < BrewConfig.MODEL_LOAD_ATTEMPTS - 1) Thread.sleep(1_000L * (attempt + 1))
                }
            }
            throw lastError ?: IllegalStateException("Whisper failed to load")
        }.get()
    }

    /** Releases both native models (call after a transcription job completes). */
    fun unload() {
        submit {
            encoder?.close()
            decoder?.close()
            encoder = null
            decoder = null
            isLoaded = false
        }.get()
    }

    /**
     * Transcribes [audio16kMono] (mono float [-1,1] @16 kHz). Slices into
     * non-overlapping 30 s windows, transcribes each, and joins with spaces.
     * Reports per-window progress via [onProgress] (0..1). A window that throws
     * is skipped with a marker; the whole job fails only if every window fails.
     */
    fun transcribe(audio16kMono: FloatArray, onProgress: (Float) -> Unit = {}): String {
        return submit {
            val windows = windowCount(audio16kMono.size)
            if (windows == 0) return@submit ""
            val parts = ArrayList<String>(windows)
            var failures = 0
            for (w in 0 until windows) {
                try {
                    val text = transcribeWindow(audio16kMono, w)
                    if (text.isNotBlank()) parts.add(text.trim())
                } catch (t: Throwable) {
                    android.util.Log.e("BrewWhisper", "window $w failed", t)
                    failures++
                    parts.add("[…part of the recording couldn't be transcribed…]")
                }
                onProgress((w + 1).toFloat() / windows)
            }
            if (failures == windows) throw IllegalStateException("Transcription failed for all windows")
            parts.joinToString(" ").trim()
        }.get()
    }

    // --- internals (all run on the engine thread) ---

    private fun transcribeWindow(audio: FloatArray, windowIndex: Int): String {
        val enc = encoder ?: error("encoder not loaded")
        val dec = decoder ?: error("decoder not loaded")
        val melFrontend = mel ?: error("mel not loaded")
        val detokenizer = detok ?: error("detokenizer not loaded")

        val start = windowIndex * WINDOW_SAMPLES
        val window = FloatArray(WINDOW_SAMPLES)
        val n = minOf(WINDOW_SAMPLES, audio.size - start)
        System.arraycopy(audio, start, window, 0, n) // remainder is zero-padded silence

        run {
            val din = dec.getInputBuffers()
            android.util.Log.i("BrewWhisper", "DEC inputs=${din.size} counts=${din.map { it.count() }} bytes=${din.map { it.size() }}")
        }
        // This encoder takes RAW 30s audio [1,480000] f32 (mel is computed inside the model).
        val encInput = Tensor.of(window, DataType.Float32, intArrayOf(1, WINDOW_SAMPLES))
        val encOut = enc.run(arrayOf(encInput))
        android.util.Log.i("BrewWhisper", "ENC out count=${encOut[0].count()} bytes=${encOut[0].size()} cap=${encOut[0].data.capacity()}")
        val encHidden = encOut[0].data
        encHidden.order(ByteOrder.LITTLE_ENDIAN)

        val hiddenDim = (encHidden.capacity() / 4) / N_AUDIO_CTX
        val ids = decodeGreedy(dec, encHidden, hiddenDim)
        android.util.Log.i("BrewWhisper", "win$windowIndex generated ${ids.size} tokens")
        return detokenizer.decode(ids)
    }

    /** Forced-English greedy decode. Returns generated (non-prefill) token ids. */
    private fun decodeGreedy(
        dec: ZeticMLangeModel,
        encHidden: ByteBuffer,
        hiddenDim: Int,
    ): IntArray {
        // Derive token-buffer length from the model itself (these custom models use a
        // non-standard length, not the 448 of the OpenAI export).
        val inputs = dec.getInputBuffers()
        val tokenLen = inputs[0].size() / 4 // int32 tokens
        val ids = IntArray(tokenLen) { PAD }
        val mask = IntArray(tokenLen)
        val prompt = intArrayOf(SOT, LANG_EN, TRANSCRIBE, NO_TIMESTAMPS)
        for (i in prompt.indices) {
            ids[i] = prompt[i]
            mask[i] = 1
        }
        var idx = prompt.size

        val encTensor = Tensor.of(encHidden, DataType.Float32, intArrayOf(encHidden.capacity() / 4))
        val generated = ArrayList<Int>()
        var lastTok = -1
        var repeats = 0
        var vocab = -1
        var logitsRow = FloatArray(0)

        while (idx < tokenLen) {
            val outputs = dec.run(
                arrayOf(
                    Tensor.of(ids, DataType.Int32, intArrayOf(tokenLen)),
                    encTensor,
                    Tensor.of(mask, DataType.Int32, intArrayOf(tokenLen)),
                )
            )
            if (vocab < 0) {
                val outFloats = outputs[0].size() / 4
                vocab = outFloats / tokenLen
                logitsRow = FloatArray(vocab)
                android.util.Log.i("BrewWhisper", "DEC out floats=$outFloats tokenLen=$tokenLen vocab=$vocab")
            }
            val logits = outputs[0].data.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            logits.position((idx - 1) * vocab)
            logits.get(logitsRow, 0, vocab)
            val next = argmax(logitsRow)
            if (next == EOT || next == PAD) break

            if (next == lastTok) {
                if (++repeats >= REPETITION_GUARD) break
            } else {
                repeats = 0
                lastTok = next
            }

            ids[idx] = next
            mask[idx] = 1
            generated.add(next)
            idx++
        }
        android.util.Log.i("BrewWhisper", "decoded ${generated.size} tokens: ${generated.take(20)}")
        return generated.toIntArray()
    }

    private fun argmax(a: FloatArray): Int {
        var best = 0
        var bestVal = a[0]
        for (i in 1 until a.size) {
            if (a[i] > bestVal) {
                bestVal = a[i]
                best = i
            }
        }
        return best
    }

    private fun <T> submit(block: () -> T): Future<T> = exec.submit(block)

    companion object {
        const val WINDOW_SAMPLES = 480_000 // 30 s @ 16 kHz
        const val N_MELS = 80
        const val N_FRAMES = 3_000
        const val N_AUDIO_CTX = 1_500
        const val MAX_LEN = 448
        const val VOCAB = 51_865

        // Whisper multilingual special token ids.
        const val SOT = 50_258          // <|startoftranscript|>
        const val LANG_EN = 50_259      // <|en|>
        const val TRANSCRIBE = 50_359   // <|transcribe|>
        const val NO_TIMESTAMPS = 50_363 // <|notimestamps|>
        const val EOT = 50_257          // <|endoftext|>
        const val PAD = 50_256

        const val REPETITION_GUARD = 24

        fun windowCount(samples: Int): Int =
            if (samples <= 0) 0 else (samples + WINDOW_SAMPLES - 1) / WINDOW_SAMPLES
    }
}
