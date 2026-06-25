package ai.zetic.aiberry.core

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Microphone recorder producing 16 kHz mono Float32 audio.
 *
 * Two modes:
 *   - auto-stop  — `record(autoStopSeconds = 3.0)` finishes itself after N seconds.
 *   - manual     — `record(autoStopSeconds = null)` records until `stop()` is called
 *                  (with a 30 s safety cap).
 *
 * Compose observes [isRecording] and [level]. Caller must hold RECORD_AUDIO permission.
 */
class AudioRecorder {
    var isRecording by mutableStateOf(false)
        private set
    var level by mutableFloatStateOf(0f)        // 0..1, drives the meter UI
        private set

    private val sampleRate = AppConfig.SAMPLE_RATE
    private val maxSamples = AppConfig.SAMPLE_RATE * 30   // hard safety cap

    private val main = Handler(Looper.getMainLooper())
    private var thread: Thread? = null
    @Volatile private var active = false

    /**
     * Start recording. Pass `autoStopSeconds = null` for tap-to-stop.
     * Safe to call repeatedly; ignored while a recording is already in flight.
     */
    @SuppressLint("MissingPermission")
    fun record(autoStopSeconds: Double?, onComplete: (FloatArray) -> Unit) {
        if (active) return
        active = true
        val autoStopSamples = autoStopSeconds?.let { (sampleRate * it).toInt() }

        thread = Thread {
            val minBytes = AudioRecord.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_FLOAT,
            )
            val bufferBytes = maxOf(minBytes, 4096 * Float.SIZE_BYTES)

            val recorder = try {
                AudioRecord(
                    MediaRecorder.AudioSource.VOICE_RECOGNITION,
                    sampleRate,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_FLOAT,
                    bufferBytes,
                )
            } catch (t: Throwable) {
                finish(null, onComplete)
                return@Thread
            }

            if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                recorder.release()
                finish(null, onComplete)
                return@Thread
            }

            val captured = ArrayList<Float>(sampleRate * 3)
            val chunk = FloatArray(2048)
            try {
                recorder.startRecording()
                main.post { isRecording = true }

                while (active) {
                    val n = recorder.read(chunk, 0, chunk.size, AudioRecord.READ_BLOCKING)
                    if (n <= 0) continue

                    var sumSq = 0f
                    for (i in 0 until n) {
                        val s = chunk[i]
                        captured.add(s)
                        sumSq += s * s
                    }
                    val rms = sqrt(sumSq / n)
                    main.post { level = min(1f, rms * 8f) }

                    if (autoStopSamples != null && captured.size >= autoStopSamples) break
                    if (captured.size >= maxSamples) break
                }
            } catch (_: Throwable) {
                // fall through to cleanup
            } finally {
                try {
                    recorder.stop()
                } catch (_: Throwable) {
                }
                recorder.release()
            }

            val all = FloatArray(captured.size) { captured[it] }
            val deliver = if (autoStopSamples != null) {
                all.copyOfRange(0, min(autoStopSamples, all.size))
            } else {
                all
            }
            finish(deliver, onComplete)
        }.also { it.start() }
    }

    /** Finish a manual recording and deliver what was captured. */
    fun stop() {
        active = false
    }

    private fun finish(deliver: FloatArray?, onComplete: (FloatArray) -> Unit) {
        active = false
        main.post {
            isRecording = false
            level = 0f
            if (deliver != null && deliver.isNotEmpty()) onComplete(deliver)
        }
    }
}
