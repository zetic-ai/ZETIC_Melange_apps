package ai.zetic.demo.posemotion.benchmark

import ai.zetic.demo.posemotion.pipeline.BenchmarkSnapshot
import ai.zetic.demo.posemotion.pipeline.FrameTimings
import android.os.SystemClock

/**
 * Rolling per-stage latency + sustained FPS + peak memory.
 * Confined to the inference thread; publish snapshot() to the UI.
 */
class BenchmarkStats {
    private val detect = RollingMean(30)
    private val pose = RollingMean(30)
    private val lift = RollingMean(30)
    private val total = RollingMean(30)
    private val frameStamps = ArrayDeque<Long>()
    private var peakMemoryMB = 0.0

    fun record(t: FrameTimings) {
        detect.push(t.detectMs)
        pose.push(t.poseMs)
        if (t.liftMs > 0) lift.push(t.liftMs)
        total.push(t.totalMs)

        val now = SystemClock.elapsedRealtimeNanos()
        frameStamps.addLast(now)
        while (frameStamps.isNotEmpty() && now - frameStamps.first() > 2_000_000_000L) {
            frameStamps.removeFirst()   // sustained FPS over a 2 s window
        }

        peakMemoryMB = maxOf(peakMemoryMB, MemoryProbe.footprintMB())
    }

    fun resetFps() = frameStamps.clear()

    fun snapshot(): BenchmarkSnapshot {
        var fps = 0.0
        if (frameStamps.size > 1) {
            val spanS = (frameStamps.last() - frameStamps.first()) / 1e9
            if (spanS > 0.2) fps = (frameStamps.size - 1) / spanS
        }
        return BenchmarkSnapshot(
            detectMs = detect.mean,
            poseMs = pose.mean,
            liftMs = lift.mean,
            totalMs = total.mean,
            fps = fps,
            memoryMB = MemoryProbe.footprintMB(),
            peakMemoryMB = peakMemoryMB,
        )
    }
}

private class RollingMean(private val capacity: Int) {
    private val values = ArrayDeque<Double>()
    fun push(v: Double) {
        values.addLast(v)
        while (values.size > capacity) values.removeFirst()
    }
    val mean: Double get() = if (values.isEmpty()) 0.0 else values.sum() / values.size
}
