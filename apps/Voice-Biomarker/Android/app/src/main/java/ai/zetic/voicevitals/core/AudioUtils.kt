package ai.zetic.voicevitals.core

import kotlin.math.ceil
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Lightweight waveform helpers to feed the emotion model cleaner input.
 *
 * Two issues hurt accuracy with real recordings (vs the model's clean, all-speech
 * training clips): (1) dead air before/after speech dilutes the model's mean-pool,
 * and (2) speech longer than the 3 s window gets cropped. These helpers trim
 * silence and split long speech into overlapping windows for logit averaging.
 */
object AudioUtils {

    /**
     * Energy-based trim of leading/trailing silence (simple VAD).
     * Keeps frames whose RMS is >= [relThreshold] x the loudest frame, plus a margin.
     */
    fun trimSilence(
        x: FloatArray,
        frame: Int = 320,            // 20 ms @ 16 kHz
        relThreshold: Float = 0.08f,
    ): FloatArray {
        if (x.size <= frame) return x

        val energies = ArrayList<Float>(x.size / frame + 1)
        var i = 0
        while (i < x.size) {
            val end = min(i + frame, x.size)
            var sum = 0f
            for (j in i until end) sum += x[j] * x[j]
            energies.add(sqrt(sum / (end - i)))
            i += frame
        }

        val maxE = energies.maxOrNull() ?: return x
        if (maxE <= 1e-6f) return x
        val thr = maxE * relThreshold
        val firstVoiced = energies.indexOfFirst { it >= thr }
        val lastVoiced = energies.indexOfLast { it >= thr }
        if (firstVoiced < 0 || lastVoiced < 0) return x

        val start = maxOf(0, (firstVoiced - 1) * frame)            // one-frame margin
        val stop = min(x.size, (lastVoiced + 2) * frame)
        if (stop <= start) return x
        return x.copyOfRange(start, stop)
    }

    /**
     * Split into up to [maxWindows] windows of [size] samples for multi-window
     * averaging. Short input returns a single window (caller pads/tiles it).
     */
    fun windows(x: FloatArray, size: Int, maxWindows: Int = 3): List<FloatArray> {
        if (x.size <= size) return listOf(x)
        val count = min(maxWindows, ceil(x.size.toDouble() / size.toDouble()).toInt())
        if (count <= 1) return listOf(x.copyOfRange(0, size))
        val stride = (x.size - size) / (count - 1)
        return (0 until count).map { x.copyOfRange(it * stride, it * stride + size) }
    }
}
