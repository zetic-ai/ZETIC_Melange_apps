package ai.zetic.demo.cameravitals.signal

import kotlin.math.abs

/** Removes a least-squares linear trend (baseline drift). Mirrors iOS Detrend.linear. */
object Detrend {
    fun linear(x: FloatArray): FloatArray {
        val n = x.size
        if (n <= 2) return x
        val nF = n.toFloat()
        var sumT = 0f; var sumT2 = 0f; var sumY = 0f; var sumTY = 0f
        for (i in 0 until n) {
            val t = i.toFloat()
            sumT += t; sumT2 += t * t; sumY += x[i]; sumTY += t * x[i]
        }
        val denom = nF * sumT2 - sumT * sumT
        if (abs(denom) < 1e-9f) return x
        val b = (nF * sumTY - sumT * sumY) / denom
        val a = (sumY - b * sumT) / nF
        return FloatArray(n) { i -> x[i] - (a + b * i) }
    }
}
