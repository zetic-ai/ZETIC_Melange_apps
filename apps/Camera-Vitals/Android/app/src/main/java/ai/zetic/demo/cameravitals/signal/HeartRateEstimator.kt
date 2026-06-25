package ai.zetic.demo.cameravitals.signal

import ai.zetic.demo.cameravitals.AppConfig
import ai.zetic.demo.cameravitals.util.clamped

/**
 * Turns a model rPPG waveform (DiffNormalized derivative) into a heart rate.
 * Mirrors rPPG-Toolbox postprocess: cumsum → detrend → bandpass → FFT peak.
 */
object HeartRateEstimator {
    class Result(val bpm: Double, val quality: Double, val filtered: FloatArray)

    fun estimate(waveform: FloatArray, fs: Double): Result? {
        if (waveform.size < 60) return null

        // Integrate (model predicts the derivative of the pulse).
        val cumulative = FloatArray(waveform.size)
        var acc = 0f
        for (i in waveform.indices) { acc += waveform[i]; cumulative[i] = acc }

        val detrended = Detrend.linear(cumulative)
        val filtered = BandpassHR.filtfilt(detrended)

        val peak = Fft.dominantBPM(filtered, fs, AppConfig.HR_BAND_LOW, AppConfig.HR_BAND_HIGH)
            ?: return null

        if (peak.bpm < AppConfig.MIN_BPM || peak.bpm > AppConfig.MAX_BPM) {
            return Result(peak.bpm, 0.0, filtered)  // physiologically implausible
        }

        // Map harmonic SNR (dB) → 0..1 quality (calibrated: clean pulse ≥0.6, noise red).
        val quality = ((peak.snr + 4.0) / 12.0).clamped(0.0, 1.0)
        return Result(peak.bpm, quality, filtered)
    }
}
