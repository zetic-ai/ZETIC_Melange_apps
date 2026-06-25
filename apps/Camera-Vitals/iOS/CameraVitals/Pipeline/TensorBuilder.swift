import Accelerate
import Foundation
import ZeticMLange

/// Builds the model input tensor from a flattened 181-frame clip, applying the
/// rPPG-Toolbox 'Standardized' preprocessing: single scalar (x - mean) / std.
enum TensorBuilder {
    static func build(_ flat: [Float]) -> Tensor? {
        let n = flat.count
        let count = vDSP_Length(n)

        // Standardize straight into the tensor's backing Data — no extra ~11 MB copy.
        var bytes = Data(count: n * MemoryLayout<Float>.size)
        var ok = false
        bytes.withUnsafeMutableBytes { raw in
            guard let dst = raw.baseAddress?.assumingMemoryBound(to: Float.self) else { return }
            var mean: Float = 0
            vDSP_meanv(flat, 1, &mean, count)
            var negMean = -mean
            vDSP_vsadd(flat, 1, &negMean, dst, 1, count)     // dst = flat - mean
            var sumsq: Float = 0
            vDSP_svesq(dst, 1, &sumsq, count)
            let std = sqrt(sumsq / Float(n))
            guard std > 1e-6 else { return }                 // flat clip → skip (avoid NaNs)
            var invStd = 1.0 / std
            vDSP_vsmul(dst, 1, &invStd, dst, 1, count)        // dst *= 1/std
            ok = true
        }
        guard ok else { return nil }

        return Tensor(
            data: bytes,
            dataType: BuiltinDataType.float32,
            shape: [AppConfig.framesIn, AppConfig.channels, AppConfig.imgSize, AppConfig.imgSize]
        )
    }
}
