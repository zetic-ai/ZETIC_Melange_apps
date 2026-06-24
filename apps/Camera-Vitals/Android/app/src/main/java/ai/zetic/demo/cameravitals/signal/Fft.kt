package ai.zetic.demo.cameravitals.signal

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.log10
import kotlin.math.log2
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

data class SpectrumPeak(val bpm: Double, val snr: Double)

/**
 * Hann-windowed, zero-padded FFT with quadratic peak interpolation + harmonic-window SNR.
 * Mirrors the iOS FFTAnalyzer (vDSP) with a hand-written radix-2 Cooley-Tukey FFT.
 */
object Fft {
    /** In-place iterative radix-2 FFT; arrays must be a power-of-two length. */
    private fun fft(real: FloatArray, imag: FloatArray) {
        val n = real.size
        // Bit-reversal permutation.
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j or bit
            if (i < j) {
                var t = real[i]; real[i] = real[j]; real[j] = t
                t = imag[i]; imag[i] = imag[j]; imag[j] = t
            }
        }
        var len = 2
        while (len <= n) {
            val ang = -2.0 * PI / len
            val wr = cos(ang).toFloat()
            val wi = sin(ang).toFloat()
            var i = 0
            while (i < n) {
                var curR = 1f; var curI = 0f
                val half = len / 2
                for (k in 0 until half) {
                    val aR = real[i + k]; val aI = imag[i + k]
                    val bR = real[i + k + half]; val bI = imag[i + k + half]
                    val tR = bR * curR - bI * curI
                    val tI = bR * curI + bI * curR
                    real[i + k] = aR + tR; imag[i + k] = aI + tI
                    real[i + k + half] = aR - tR; imag[i + k + half] = aI - tI
                    val nR = curR * wr - curI * wi
                    curI = curR * wi + curI * wr
                    curR = nR
                }
                i += len
            }
            len = len shl 1
        }
    }

    fun dominantBPM(signal: FloatArray, fs: Double, lo: Double, hi: Double): SpectrumPeak? {
        val n = signal.size
        if (n < 16) return null

        val fftN = 1 shl ceil(log2(n.toDouble())).toInt()
        val real = FloatArray(fftN)
        val imag = FloatArray(fftN)
        // Hann window over the populated samples (matches numpy.hanning used in validation).
        for (i in 0 until n) {
            val w = 0.5f * (1f - cos(2.0 * PI * i / (n - 1)).toFloat())
            real[i] = signal[i] * w
        }
        fft(real, imag)

        val half = fftN / 2
        val mag = FloatArray(half) { sqrt(real[it] * real[it] + imag[it] * imag[it]) }

        val loBin = ceil(lo * fftN / fs).toInt()
        val hiBin = floor(hi * fftN / fs).toInt()
        if (loBin < 1 || hiBin >= half || hiBin <= loBin) return null

        var peakBin = loBin
        var peakVal = mag[loBin]
        var bandPower = 0f
        for (k in loBin..hiBin) {
            bandPower += mag[k] * mag[k]
            if (mag[k] > peakVal) { peakVal = mag[k]; peakBin = k }
        }

        var interp = peakBin.toDouble()
        if (peakBin > loBin && peakBin < hiBin) {
            val a = mag[peakBin - 1].toDouble()
            val b = mag[peakBin].toDouble()
            val c = mag[peakBin + 1].toDouble()
            val denom = a - 2 * b + c
            if (abs(denom) > 1e-9) interp += 0.5 * (a - c) / denom
        }

        val f0 = fs * interp / fftN
        val bpm = f0 * 60.0

        // SNR = energy at fundamental + 2nd harmonic vs the rest of the band.
        val halfWin = 0.15
        var signalPower = 0f
        for (k in loBin..hiBin) {
            val f = fs * k / fftN
            if (abs(f - f0) <= halfWin || abs(f - 2 * f0) <= halfWin) signalPower += mag[k] * mag[k]
        }
        val noisePower = max(bandPower - signalPower, 1e-9f)
        val snr = if (signalPower > 0) 10 * log10(signalPower.toDouble() / noisePower.toDouble()) else -20.0
        return SpectrumPeak(bpm, snr)
    }
}
