import Accelerate
import CoreGraphics
import CoreVideo

struct CroppedFrame {
    let planarRGB: [Float]   // length 3*size*size, CHW (R plane, G plane, B plane), values 0-255
    let meanLuma: Float
}

/// Crops a face ROI from a BGRA pixel buffer and resizes to `size`×`size`,
/// emitting planar RGB float — exactly what the model input tensor needs.
enum FrameCropper {
    static func crop(_ pixelBuffer: CVPixelBuffer, roi: CGRect, size: Int = AppConfig.imgSize) -> CroppedFrame? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let bpr = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)

        // Integer ROI clamped to the frame.
        let x = Int(roi.origin.x).clamped(to: 0...(w - 1))
        let y = Int(roi.origin.y).clamped(to: 0...(h - 1))
        let rw = Int(roi.width).clamped(to: 1...(w - x))
        let rh = Int(roi.height).clamped(to: 1...(h - y))

        var srcBuf = vImage_Buffer(
            data: base.advanced(by: y * bpr + x * 4),
            height: vImagePixelCount(rh),
            width: vImagePixelCount(rw),
            rowBytes: bpr
        )

        let destRowBytes = size * 4
        var destData = [UInt8](repeating: 0, count: size * destRowBytes)
        let err: vImage_Error = destData.withUnsafeMutableBytes { ptr in
            var destBuf = vImage_Buffer(
                data: ptr.baseAddress,
                height: vImagePixelCount(size),
                width: vImagePixelCount(size),
                rowBytes: destRowBytes
            )
            return vImageScale_ARGB8888(&srcBuf, &destBuf, nil, vImage_Flags(kvImageHighQualityResampling))
        }
        guard err == kvImageNoError else { return nil }

        // Deinterleave BGRA → planar RGB float (drop alpha). BGRA memory order: B,G,R,A.
        let n = size * size
        var planar = [Float](repeating: 0, count: 3 * n)
        var lumaSum: Float = 0
        for i in 0..<n {
            let b = Float(destData[i * 4 + 0])
            let g = Float(destData[i * 4 + 1])
            let r = Float(destData[i * 4 + 2])
            planar[i] = r              // R plane
            planar[n + i] = g          // G plane
            planar[2 * n + i] = b      // B plane
            lumaSum += 0.299 * r + 0.587 * g + 0.114 * b
        }
        return CroppedFrame(planarRGB: planar, meanLuma: lumaSum / Float(n))
    }
}
