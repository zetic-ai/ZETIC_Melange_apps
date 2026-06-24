import Foundation

/// Removes a least-squares linear trend (baseline drift) from a signal.
/// Combined with the zero-phase bandpass downstream, this matches the
/// peak-frequency behaviour of the toolbox's smoothness-prior detrend.
enum Detrend {
    static func linear(_ x: [Float]) -> [Float] {
        let n = x.count
        guard n > 2 else { return x }
        let nF = Float(n)
        var sumT: Float = 0, sumT2: Float = 0, sumY: Float = 0, sumTY: Float = 0
        for i in 0..<n {
            let t = Float(i)
            sumT += t; sumT2 += t * t; sumY += x[i]; sumTY += t * x[i]
        }
        let denom = nF * sumT2 - sumT * sumT
        guard abs(denom) > 1e-9 else { return x }
        let b = (nF * sumTY - sumT * sumY) / denom
        let a = (sumY - b * sumT) / nF
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n { out[i] = x[i] - (a + b * Float(i)) }
        return out
    }
}
