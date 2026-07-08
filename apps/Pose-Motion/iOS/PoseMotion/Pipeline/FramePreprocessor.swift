import Accelerate
import CoreImage
import CoreVideo
import Foundation
import ZeticMLange

/// CVPixelBuffer (BGRA) → model input tensors, with all buffers preallocated.
/// Confined to the inference queue.
final class FramePreprocessor {
    private let detSize = AppConfig.detSize
    private let poseW = AppConfig.poseInputWidth
    private let poseH = AppConfig.poseInputHeight

    // Reusable buffers
    private var detRGBA: [UInt8]
    private var detFloats: [Float]
    private var poseRGBA: [UInt8]
    private var poseFloats: [Float]
    private let ciContext = CIContext(options: [.cacheIntermediates: false])

    init() {
        detRGBA = [UInt8](repeating: 0, count: detSize * detSize * 4)
        detFloats = [Float](repeating: 0, count: detSize * detSize * 3)
        poseRGBA = [UInt8](repeating: 0, count: poseW * poseH * 4)
        poseFloats = [Float](repeating: 0, count: poseW * poseH * 3)
    }

    /// Full frame → [1,3,640,640] RGB 0..1 tensor (stretch resize; the detector's
    /// normalized outputs are mapped straight back to the frame, so the stretch cancels).
    func detectorTensor(from pixelBuffer: CVPixelBuffer) -> Tensor? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let srcBase = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }

        var src = vImage_Buffer(
            data: srcBase,
            height: vImagePixelCount(CVPixelBufferGetHeight(pixelBuffer)),
            width: vImagePixelCount(CVPixelBufferGetWidth(pixelBuffer)),
            rowBytes: CVPixelBufferGetBytesPerRow(pixelBuffer)
        )
        scaleAndPack(&src, into: &detRGBA, width: detSize, height: detSize, floats: &detFloats)
        let data = detFloats.withUnsafeBufferPointer { Data(buffer: $0) }
        return Tensor(data: data, dataType: BuiltinDataType.float32, shape: [1, 3, detSize, detSize])
    }

    /// Person crop (pixel rect within the frame) → [1,3,256,192] RGB 0..1 tensor.
    func poseTensor(from pixelBuffer: CVPixelBuffer, cropPixels: CGRect) -> Tensor? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let srcBase = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }

        let frameW = CVPixelBufferGetWidth(pixelBuffer)
        let frameH = CVPixelBufferGetHeight(pixelBuffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)

        let x = max(0, min(frameW - 2, Int(cropPixels.minX)))
        let y = max(0, min(frameH - 2, Int(cropPixels.minY)))
        let w = max(2, min(frameW - x, Int(cropPixels.width)))
        let h = max(2, min(frameH - y, Int(cropPixels.height)))

        var src = vImage_Buffer(
            data: srcBase.advanced(by: y * rowBytes + x * 4),
            height: vImagePixelCount(h),
            width: vImagePixelCount(w),
            rowBytes: rowBytes
        )
        scaleAndPack(&src, into: &poseRGBA, width: poseW, height: poseH, floats: &poseFloats)
        let data = poseFloats.withUnsafeBufferPointer { Data(buffer: $0) }
        return Tensor(data: data, dataType: BuiltinDataType.float32, shape: [1, 3, poseH, poseW])
    }

    /// Frame → CGImage for display (reused CIContext).
    func displayImage(from pixelBuffer: CVPixelBuffer) -> CGImage? {
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        return ciContext.createCGImage(ci, from: ci.extent)
    }

    // MARK: - Private

    /// vImage scale into a reusable RGBA byte buffer, then BGRA → planar RGB float 0..1.
    private func scaleAndPack(_ src: inout vImage_Buffer, into rgba: inout [UInt8],
                              width: Int, height: Int, floats: inout [Float]) {
        rgba.withUnsafeMutableBufferPointer { dstPtr in
            guard let dstBase = dstPtr.baseAddress else { return }
            var dst = vImage_Buffer(
                data: dstBase,
                height: vImagePixelCount(height),
                width: vImagePixelCount(width),
                rowBytes: width * 4
            )
            vImageScale_ARGB8888(&src, &dst, nil, vImage_Flags(kvImageNoFlags))
        }
        rgba.withUnsafeBufferPointer { pixelPtr in
            floats.withUnsafeMutableBufferPointer { floatPtr in
                guard let s = pixelPtr.baseAddress, let d = floatPtr.baseAddress else { return }
                let count = width * height
                for i in 0..<count {
                    // Source is BGRA (camera/AVAssetReader default); dest is planar RGB.
                    let offset = i * 4
                    d[i] = Float(s[offset + 2]) / 255.0
                    d[count + i] = Float(s[offset + 1]) / 255.0
                    d[2 * count + i] = Float(s[offset + 0]) / 255.0
                }
            }
        }
    }
}
