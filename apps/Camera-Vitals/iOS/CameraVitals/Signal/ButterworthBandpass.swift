import Foundation

/// Direct-Form-II transposed biquad (one 2nd-order section).
struct Biquad {
    let b: [Float]   // b0, b1, b2
    let a: [Float]   // 1, a1, a2

    func apply(_ x: [Float]) -> [Float] {
        var y = [Float](repeating: 0, count: x.count)
        var x1: Float = 0, x2: Float = 0, y1: Float = 0, y2: Float = 0
        for i in 0..<x.count {
            let xi = x[i]
            let yi = b[0] * xi + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
            y[i] = yi
            x2 = x1; x1 = xi
            y2 = y1; y1 = yi
        }
        return y
    }
}

/// Zero-phase Butterworth bandpass (0.75–2.5 Hz @ 30 fps).
/// Coefficients computed offline with scipy `butter(1, [0.75,2.5]/fs*2, 'bandpass')`
/// and validated end-to-end in scripts/validate_pipeline.py.
enum BandpassHR {
    static let biquad = Biquad(
        b: [0.1563595206991934, 0.0, -0.1563595206991934],
        a: [1.0, -1.6175876941699503, 0.6872809586016133]
    )

    /// filtfilt: forward then reverse-time pass for zero phase distortion.
    static func filtfilt(_ x: [Float]) -> [Float] {
        guard x.count > 6 else { return x }
        let fwd = biquad.apply(x)
        let rev = biquad.apply(Array(fwd.reversed()))
        return Array(rev.reversed())
    }
}
