import Accelerate
import CoreGraphics
import Foundation
import ZeticMLange

struct Keypoint2D {
    var x: CGFloat          // normalized 0..1 in frame coordinates
    var y: CGFloat
    var conf: Float
}

/// Decodes RTMPose SimCC outputs: per joint, argmax over the x/y bin rows,
/// divided by the split ratio → crop-pixel coords → frame coords.
/// Confidence = min of the two softmax peak probabilities (the outputs are logits).
final class SimCCDecoder {
    private var scratch = [Float](repeating: 0, count: 1024)

    /// `simccX` [1,17,Wbins], `simccY` [1,17,Hbins]; `cropPixels` is the pose crop in frame pixels.
    func decode(simccX: Tensor, simccY: Tensor,
                cropPixels: CGRect, frameSize: CGSize) -> [Keypoint2D] {
        let joints = AppConfig.jointCount
        let xBins = simccX.shape.last ?? 0
        let yBins = simccY.shape.last ?? 0
        let xData: [Float] = simccX.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let yData: [Float] = simccY.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        guard xData.count >= joints * xBins, yData.count >= joints * yBins else { return [] }

        let split = CGFloat(AppConfig.simccSplitRatio)
        let cropW = CGFloat(AppConfig.poseInputWidth)
        let cropH = CGFloat(AppConfig.poseInputHeight)

        var result: [Keypoint2D] = []
        result.reserveCapacity(joints)
        for j in 0..<joints {
            let (xIdx, xConf) = peak(xData, offset: j * xBins, count: xBins)
            let (yIdx, yConf) = peak(yData, offset: j * yBins, count: yBins)

            let xCrop = CGFloat(xIdx) / split      // 0..192 crop-pixel space
            let yCrop = CGFloat(yIdx) / split      // 0..256
            let frameX = cropPixels.minX + xCrop / cropW * cropPixels.width
            let frameY = cropPixels.minY + yCrop / cropH * cropPixels.height

            result.append(Keypoint2D(
                x: frameX / frameSize.width,
                y: frameY / frameSize.height,
                conf: min(xConf, yConf)
            ))
        }
        return result
    }

    /// Returns (argmax index, softmax peak probability) for one bin row.
    private func peak(_ data: [Float], offset: Int, count: Int) -> (Int, Float) {
        var maxValue: Float = 0
        var maxIndex: vDSP_Length = 0
        data.withUnsafeBufferPointer { ptr in
            vDSP_maxvi(ptr.baseAddress! + offset, 1, &maxValue, &maxIndex, vDSP_Length(count))
        }
        // softmax peak = 1 / sum(exp(v - max))
        var negMax = -maxValue
        data.withUnsafeBufferPointer { ptr in
            vDSP_vsadd(ptr.baseAddress! + offset, 1, &negMax, &scratch, 1, vDSP_Length(count))
        }
        var n = Int32(count)
        vvexpf(&scratch, scratch, &n)
        var sum: Float = 0
        vDSP_sve(scratch, 1, &sum, vDSP_Length(count))
        return (Int(maxIndex), sum > 0 ? 1.0 / sum : 0)
    }
}
