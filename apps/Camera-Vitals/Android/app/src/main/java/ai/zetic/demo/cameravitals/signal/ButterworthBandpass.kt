package ai.zetic.demo.cameravitals.signal

/** Direct-Form-II transposed biquad (one 2nd-order section). */
class Biquad(private val b: FloatArray, private val a: FloatArray) {
    fun apply(x: FloatArray): FloatArray {
        val y = FloatArray(x.size)
        var x1 = 0f; var x2 = 0f; var y1 = 0f; var y2 = 0f
        for (i in x.indices) {
            val xi = x[i]
            val yi = b[0] * xi + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
            y[i] = yi
            x2 = x1; x1 = xi
            y2 = y1; y1 = yi
        }
        return y
    }
}

/**
 * Zero-phase Butterworth bandpass (0.75–2.5 Hz @ 30 fps). Coefficients computed offline with
 * scipy `butter(1, [0.75,2.5]/fs*2, 'bandpass')` — identical to the iOS app and validated
 * end-to-end in scripts/validate_stitch.py.
 */
object BandpassHR {
    private val biquad = Biquad(
        floatArrayOf(0.1563595206991934f, 0.0f, -0.1563595206991934f),
        floatArrayOf(1.0f, -1.6175876941699503f, 0.6872809586016133f)
    )

    /** filtfilt: forward then reverse-time pass for zero phase distortion. */
    fun filtfilt(x: FloatArray): FloatArray {
        if (x.size <= 6) return x
        val fwd = biquad.apply(x)
        val rev = biquad.apply(fwd.reversedArray())
        return rev.reversedArray()
    }
}
