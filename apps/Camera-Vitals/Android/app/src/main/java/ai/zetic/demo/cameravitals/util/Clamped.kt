package ai.zetic.demo.cameravitals.util

fun Double.clamped(min: Double, max: Double): Double = if (this < min) min else if (this > max) max else this
fun Float.clamped(min: Float, max: Float): Float = if (this < min) min else if (this > max) max else this
fun Int.clamped(min: Int, max: Int): Int = if (this < min) min else if (this > max) max else this
