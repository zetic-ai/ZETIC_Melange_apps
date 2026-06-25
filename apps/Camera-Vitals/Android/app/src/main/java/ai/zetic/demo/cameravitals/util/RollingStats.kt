package ai.zetic.demo.cameravitals.util

/**
 * Median window followed by an EMA — keeps the displayed BPM steady so a single
 * noisy window never flashes a wild number on screen. (Mirrors iOS MedianEMA.)
 */
class MedianEMA(private val size: Int = 5, private val alpha: Double = 0.3) {
    private val window = ArrayDeque<Double>()
    var value: Double? = null
        private set

    fun update(x: Double): Double {
        window.addLast(x)
        if (window.size > size) window.removeFirst()
        val med = window.sorted()[window.size / 2]
        val v = value
        value = if (v != null) alpha * med + (1 - alpha) * v else med
        return value!!
    }

    fun reset() {
        window.clear()
        value = null
    }
}
