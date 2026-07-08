import Foundation
import QuartzCore

struct FrameTimings {
    var detectMs: Double = 0
    var poseMs: Double = 0
    var liftMs: Double = 0
    var totalMs: Double = 0
}

struct BenchmarkSnapshot {
    var detectMs: Double = 0
    var poseMs: Double = 0
    var liftMs: Double = 0
    var totalMs: Double = 0
    var fps: Double = 0
    var memoryMB: Double = 0
    var peakMemoryMB: Double = 0
}

/// Rolling per-stage latency + sustained FPS + peak memory.
/// Confined to the inference queue; publish `snapshot()` to the UI.
final class BenchmarkStats {
    private var detect = RollingMean(capacity: 30)
    private var pose = RollingMean(capacity: 30)
    private var lift = RollingMean(capacity: 30)
    private var total = RollingMean(capacity: 30)
    private var frameStamps: [CFTimeInterval] = []
    private var peakMemoryMB: Double = 0

    func record(_ t: FrameTimings) {
        detect.push(t.detectMs)
        pose.push(t.poseMs)
        if t.liftMs > 0 { lift.push(t.liftMs) }
        total.push(t.totalMs)

        let now = CACurrentMediaTime()
        frameStamps.append(now)
        frameStamps.removeAll { now - $0 > 2.0 }   // sustained FPS over a 2 s window

        peakMemoryMB = max(peakMemoryMB, MemoryProbe.footprintMB())
    }

    func snapshot() -> BenchmarkSnapshot {
        var s = BenchmarkSnapshot()
        s.detectMs = detect.mean
        s.poseMs = pose.mean
        s.liftMs = lift.mean
        s.totalMs = total.mean
        if let first = frameStamps.first, frameStamps.count > 1 {
            let span = CACurrentMediaTime() - first
            if span > 0.2 { s.fps = Double(frameStamps.count - 1) / span }
        }
        s.memoryMB = MemoryProbe.footprintMB()
        s.peakMemoryMB = peakMemoryMB
        return s
    }
}

private struct RollingMean {
    private var values: [Double] = []
    private let capacity: Int
    init(capacity: Int) { self.capacity = capacity }
    mutating func push(_ v: Double) {
        values.append(v)
        if values.count > capacity { values.removeFirst() }
    }
    var mean: Double { values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count) }
}
