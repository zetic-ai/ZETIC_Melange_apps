package com.brew.asr

import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sin

/**
 * Pure-Kotlin Whisper log-mel spectrogram, ported 1:1 from VoxScribe's
 * `log_mel.dart` (which reproduces openai-whisper `log_mel_spectrogram` exactly).
 *
 * The ZeticMLange 1.6.1 SDK does not expose its native `WhisperWrapper` mel, so
 * we compute it here. The 80x201 Slaney filterbank is bundled as the asset
 * `mel_filters_80.bin` (== librosa.filters.mel(16000, 400, 80)).
 *
 *   stft(n_fft=400, hop=160, periodic hann, center reflect-pad n_fft/2);
 *   power = |stft[..., :-1]|^2 (drop last frame); mel = filters @ power;
 *   log10(clamp 1e-10); clamp to (max - 8); then (x + 4) / 4.
 */
class LogMel(private val filters: FloatArray) {

    init {
        require(filters.size == N_MELS * N_FREQS) {
            "filterbank must be ${N_MELS * N_FREQS} floats, got ${filters.size}"
        }
    }

    // Periodic Hann window (matches torch hann_window default).
    private val window = DoubleArray(N_FFT) { n -> 0.5 - 0.5 * cos(2 * Math.PI * n / N_FFT) }

    // DFT twiddle tables for the 201 retained bins (direct DFT; n_fft=400 is not
    // a power of two, correctness over an FFT).
    private val cosT = Array(N_FREQS) { k ->
        DoubleArray(N_FFT) { n -> cos(-2 * Math.PI * k * n / N_FFT) }
    }
    private val sinT = Array(N_FREQS) { k ->
        DoubleArray(N_FFT) { n -> sin(-2 * Math.PI * k * n / N_FFT) }
    }

    /**
     * Computes the log-mel for [audio] and returns it as a row-major
     * `[80, frames]` FloatArray (index = mel*frames + frame). For a 480000-sample
     * (30 s) window this yields frames = 3000, ready as `[1, 80, 3000]`.
     */
    fun compute(audio: FloatArray): FloatArray {
        val pad = N_FFT / 2
        val padded = reflectPad(audio, pad)
        val frames = 1 + (padded.size - N_FFT) / HOP - 1

        val power = DoubleArray(N_FREQS)
        val logRow = DoubleArray(N_MELS * frames)
        var gMax = -1e30

        val zeroLog = -10.0 // log10(1e-10): the value an all-zero (silence) frame collapses to
        val seg = DoubleArray(N_FFT)

        for (t in 0 until frames) {
            val s = t * HOP
            var nonZero = false
            for (n in 0 until N_FFT) {
                if (padded[s + n] != 0.0) {
                    nonZero = true
                    break
                }
            }
            if (!nonZero) {
                for (m in 0 until N_MELS) logRow[m * frames + t] = zeroLog
                if (zeroLog > gMax) gMax = zeroLog
                continue
            }
            for (n in 0 until N_FFT) seg[n] = padded[s + n] * window[n]
            for (k in 0 until N_FREQS) {
                val ck = cosT[k]
                val sk = sinT[k]
                var re = 0.0
                var im = 0.0
                for (n in 0 until N_FFT) {
                    val v = seg[n]
                    re += v * ck[n]
                    im += v * sk[n]
                }
                power[k] = re * re + im * im
            }
            for (m in 0 until N_MELS) {
                val base = m * N_FREQS
                var acc = 0.0
                for (k in 0 until N_FREQS) acc += filters[base + k] * power[k]
                if (acc < 1e-10) acc = 1e-10
                val lv = log10(acc)
                logRow[m * frames + t] = lv
                if (lv > gMax) gMax = lv
            }
        }

        val floor = gMax - 8.0
        val out = FloatArray(N_MELS * frames)
        for (i in out.indices) {
            var lv = logRow[i]
            if (lv < floor) lv = floor
            out[i] = ((lv + 4.0) / 4.0).toFloat()
        }
        return out
    }

    private fun reflectPad(a: FloatArray, p: Int): DoubleArray {
        val n = a.size
        val out = DoubleArray(n + 2 * p)
        for (i in 0 until p) out[i] = a[p - i].toDouble()          // a[p], a[p-1], ..., a[1]
        for (i in 0 until n) out[p + i] = a[i].toDouble()
        for (i in 0 until p) out[p + n + i] = a[n - 2 - i].toDouble() // a[n-2], a[n-3], ...
        return out
    }

    companion object {
        const val N_FFT = 400
        const val HOP = 160
        const val N_MELS = 80
        const val N_FREQS = N_FFT / 2 + 1 // 201

        private val LN10 = ln(10.0)
        private fun log10(x: Double) = ln(x) / LN10

        /** Loads the bundled 80x201 little-endian float32 Slaney filterbank. */
        fun fromAsset(context: Context, assetName: String = "mel_filters_80.bin"): LogMel {
            val bytes = context.assets.open(assetName).use { it.readBytes() }
            val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val floats = FloatArray(bytes.size / 4) { bb.getFloat(it * 4) }
            return LogMel(floats)
        }
    }
}
