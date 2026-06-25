import Foundation

/// Packs an `size × size` RGBA8 face crop into the hosted FER model's input layout:
/// `1 × 3 × size × size`, channel-major (CHW), **BGR**, raw 0–255 with per-channel
/// mean subtraction (no /255, no std) — matching Elena Ryumina's `PreprocessInput`
/// (RGB→BGR flip, subtract [91.4953, 103.8827, 131.0912]). The model expects the
/// caller to normalize; ZETIC did not bake it into the graph.
enum FacePixelTensor {
    static func bgrMeanSubtracted(_ rgba: [UInt8], size: Int) -> [Float] {
        let n = size * size
        let (mb, mg, mr) = AppConfig.Face.bgrMean
        var out = [Float](repeating: 0, count: 3 * n)
        out.withUnsafeMutableBufferPointer { o in
            rgba.withUnsafeBufferPointer { src in
                for i in 0..<n {
                    let p = i * 4
                    o[i]         = Float(src[p + 2]) - mb   // B plane (mean-subtracted)
                    o[n + i]     = Float(src[p + 1]) - mg   // G plane
                    o[2 * n + i] = Float(src[p])     - mr   // R plane
                }
            }
        }
        return out
    }
}
