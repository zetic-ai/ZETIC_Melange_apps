import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import ZeticMLange

struct FrameResult {
    let image: CGImage?
    let frameSize: CGSize
    let personBox: CGRect?          // normalized
    let ballBox: CGRect?            // normalized
    let keypoints: [Keypoint2D]?    // COCO-17, normalized frame coords
    let pose3D: [SIMD3<Float>]?     // H36M-17, root-relative
    let timings: FrameTimings
}

/// The three-model chain: YOLO26n (person + ball) → RTMPose (crop → 2D skeleton)
/// → MotionBERT-Lite (keypoint window → 3D). Everything runs on one serial queue.
final class PosePipeline {
    let queue = DispatchQueue(label: "ai.zetic.posemotion.inference")
    let detector = MelangeModel(label: "YOLO26n")
    let pose = MelangeModel(label: "RTMPose-s")
    let lift = MelangeModel(label: "MotionBERT-Lite")

    /// Fires on the main thread with the latest processed frame.
    var onResult: ((FrameResult, BenchmarkSnapshot) -> Void)?

    private let preprocessor = FramePreprocessor()
    private let simcc = SimCCDecoder()
    private let window = KeypointWindow()
    private let stats = BenchmarkStats()
    private var frameCounter = 0
    private var lastKeypoints: [Keypoint2D]?
    private var lastPose3D: [SIMD3<Float>]?

    func resetSession() {
        queue.async { [weak self] in
            self?.window.reset()
            self?.frameCounter = 0
            self?.lastKeypoints = nil
            self?.lastPose3D = nil
        }
    }

    /// Must be called on `queue` (the ClipFrameSource pump already is).
    func process(_ pixelBuffer: CVPixelBuffer, at time: CMTime) {
        let t0 = CFAbsoluteTimeGetCurrent()
        var timings = FrameTimings()
        let frameSize = CGSize(
            width: CVPixelBufferGetWidth(pixelBuffer),
            height: CVPixelBufferGetHeight(pixelBuffer)
        )

        // 1. Detector: person + ball
        var person: Detection?
        var ball: Detection?
        if detector.isLoaded, let input = preprocessor.detectorTensor(from: pixelBuffer) {
            if let outputs = try? detector.run([input]), let out = outputs.first {
                (person, ball) = DetectorDecoder.decode(out)
                timings.detectMs = detector.lastLatencyMs
            }
        }

        // 2. Pose: crop → SimCC → keypoints
        var keypoints: [Keypoint2D]?
        if pose.isLoaded, let personBox = person?.rect {
            let crop = PersonCropper.cropRect(for: personBox, frameSize: frameSize)
            if crop.width > 4, crop.height > 4,
               let input = preprocessor.poseTensor(from: pixelBuffer, cropPixels: crop),
               let outputs = try? pose.run([input]), outputs.count >= 2 {
                // Identify simcc_x vs simcc_y by bin count, not output order.
                let a = outputs[0], b = outputs[1]
                let aBins = a.shape.last ?? 0
                let bBins = b.shape.last ?? 0
                let (sx, sy) = aBins <= bBins ? (a, b) : (b, a)   // 384 x-bins < 512 y-bins
                keypoints = simcc.decode(simccX: sx, simccY: sy,
                                         cropPixels: crop, frameSize: frameSize)
                timings.poseMs = pose.lastLatencyMs
            }
        }

        // 3. Keypoint window (repeat last frame on a miss so the lift window stays coherent)
        if let kps = keypoints {
            window.push(kps, frameSize: frameSize)
            lastKeypoints = kps
        } else if let last = lastKeypoints {
            window.push(last.map { Keypoint2D(x: $0.x, y: $0.y, conf: 0.1) }, frameSize: frameSize)
        }

        // 4. 3D lift, every liftStride frames
        if lift.isLoaded, frameCounter % AppConfig.liftStride == 0,
           window.hasData, let input = window.tensorData() {
            if let outputs = try? lift.run([input]), let out = outputs.first {
                lastPose3D = Self.readPose3D(out)
                timings.liftMs = lift.lastLatencyMs
            }
        }
        frameCounter += 1

        timings.totalMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        stats.record(timings)

        let result = FrameResult(
            image: preprocessor.displayImage(from: pixelBuffer),
            frameSize: frameSize,
            personBox: person?.rect,
            ballBox: ball?.rect,
            keypoints: keypoints,
            pose3D: lastPose3D,
            timings: timings
        )
        let snapshot = stats.snapshot()
        DispatchQueue.main.async { [weak self] in
            self?.onResult?(result, snapshot)
        }
    }

    /// [1,T,17,3] → the configured window frame as 17 root-relative 3D joints.
    private static func readPose3D(_ output: Tensor) -> [SIMD3<Float>]? {
        let shape = output.shape
        guard shape.count == 4, shape[2] == 17, shape[3] == 3 else { return nil }
        let t = min(AppConfig.liftReadIndex, shape[1] - 1)
        let floats: [Float] = output.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let base = t * 17 * 3
        guard floats.count >= base + 17 * 3 else { return nil }
        return (0..<17).map { j in
            SIMD3(floats[base + j * 3], floats[base + j * 3 + 1], floats[base + j * 3 + 2])
        }
    }
}
