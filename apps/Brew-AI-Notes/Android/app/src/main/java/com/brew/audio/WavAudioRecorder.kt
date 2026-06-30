package com.brew.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.File
import kotlin.concurrent.thread
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Captures the meeting at 16 kHz mono float PCM (like whisper-tiny's
 * `AudioSampler`) and streams it to a PCM16 WAV file on disk. Publishes a
 * smoothed 0..1 RMS level for the recording UI.
 *
 * Disk-streaming (not an in-RAM ArrayList) is deliberate: a 60-min meeting is
 * ~230 MB of float samples and would OOM if buffered.
 */
class WavAudioRecorder(
    private val outputFile: File,
    private val onLevel: (Float) -> Unit,
    private val onElapsed: (Int) -> Unit,
) {
    private val sampleRate = WavIo.TARGET_SAMPLE_RATE

    @Volatile private var recording = false
    @Volatile private var paused = false
    private var worker: Thread? = null
    private var startError: Throwable? = null

    private var smoothedLevel = 0f
    @Volatile var samplesWritten = 0L
        private set

    val durationSeconds: Int
        get() = (samplesWritten / sampleRate).toInt()

    @SuppressLint("MissingPermission")
    fun start() {
        if (recording) return
        recording = true
        paused = false
        worker = thread(name = "brew-recorder") {
            val minBuf = AudioRecord.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_FLOAT,
            )
            val bufSize = if (minBuf > 0) minBuf * 2 else sampleRate * 2
            val record = try {
                AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    sampleRate,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_FLOAT,
                    bufSize,
                )
            } catch (t: Throwable) {
                startError = t
                recording = false
                return@thread
            }
            if (record.state != AudioRecord.STATE_INITIALIZED) {
                startError = IllegalStateException("AudioRecord failed to initialize")
                recording = false
                record.release()
                return@thread
            }

            val temp = FloatArray(min(bufSize, 4096))
            val writer = WavIo.Writer(outputFile, sampleRate)
            try {
                record.startRecording()
                while (recording) {
                    val read = record.read(temp, 0, temp.size, AudioRecord.READ_BLOCKING)
                    if (read <= 0) continue
                    if (paused) {
                        emitLevel(0f)
                        continue
                    }
                    writer.write(temp, read)
                    samplesWritten += read
                    emitLevel(rms(temp, read))
                    onElapsed(durationSeconds)
                }
            } catch (_: Throwable) {
                // Stop requested or device error; WAV is finalized in finally.
            } finally {
                try {
                    record.stop()
                } catch (_: Throwable) {
                }
                record.release()
                writer.close()
            }
        }
    }

    fun pause() {
        paused = true
    }

    fun resume() {
        paused = false
    }

    /** Stops capture and returns the recorded duration in seconds. */
    fun stop(): Int {
        recording = false
        worker?.join(2_000)
        worker = null
        return durationSeconds
    }

    val error: Throwable? get() = startError

    private fun emitLevel(raw: Float) {
        // Smoothing matches iOS AudioRecorder.updateLevel.
        smoothedLevel = smoothedLevel * 0.6f + raw * 0.4f
        onLevel(smoothedLevel)
    }

    private fun rms(buf: FloatArray, count: Int): Float {
        var sum = 0.0
        for (i in 0 until count) {
            val v = buf[i]
            sum += v * v
        }
        val rms = sqrt(sum / count).toFloat()
        return (rms * 20f).coerceIn(0f, 1f)
    }
}
