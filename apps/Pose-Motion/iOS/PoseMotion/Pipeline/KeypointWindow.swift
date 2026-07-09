import CoreGraphics
import Foundation
import ZeticMLange

/// Sliding window of the last T frames of 2D keypoints, converted COCO-17 → H36M-17
/// and normalized for MotionBERT ([1,T,17,3] with x,y centered and divided by
/// min(W,H)/2). Confined to the inference queue.
final class KeypointWindow {
    private let T = AppConfig.liftWindow
    private var frames: [[Float]] = []       // each is 17*3 floats, chronological

    var hasData: Bool { !frames.isEmpty }

    func reset() {
        frames.removeAll()
    }

    /// `coco` is 17 keypoints in COCO order, normalized 0..1 frame coordinates.
    func push(_ coco: [Keypoint2D], frameSize: CGSize) {
        guard coco.count == 17 else { return }
        let h36m = Self.cocoToH36M(coco)
        // MotionBERT image normalization: (px - W/2) / (min(W,H)/2), same for y.
        let w = Float(frameSize.width)
        let h = Float(frameSize.height)
        let halfMin = min(w, h) / 2

        var frame = [Float](repeating: 0, count: 17 * 3)
        for (i, kp) in h36m.enumerated() {
            frame[i * 3] = (Float(kp.x) * w - w / 2) / halfMin
            frame[i * 3 + 1] = (Float(kp.y) * h - h / 2) / halfMin
            frame[i * 3 + 2] = kp.conf
        }
        frames.append(frame)
        if frames.count > T { frames.removeFirst(frames.count - T) }
    }

    /// [1,T,17,3] tensor, left-padded by repeating the oldest frame until T real frames exist.
    func tensorData() -> Tensor? {
        guard let first = frames.first else { return nil }
        var flat = [Float]()
        flat.reserveCapacity(T * 17 * 3)
        for _ in 0..<(T - frames.count) { flat.append(contentsOf: first) }
        for f in frames { flat.append(contentsOf: f) }
        let data = flat.withUnsafeBufferPointer { Data(buffer: $0) }
        return Tensor(data: data, dataType: BuiltinDataType.float32, shape: [1, T, 17, 3])
    }

    /// Standard COCO-17 → H36M-17 reorder (midpoints average position and confidence).
    static func cocoToH36M(_ c: [Keypoint2D]) -> [Keypoint2D] {
        func mid(_ a: Keypoint2D, _ b: Keypoint2D) -> Keypoint2D {
            Keypoint2D(x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, conf: (a.conf + b.conf) / 2)
        }
        let pelvis = mid(c[11], c[12])
        let thorax = mid(c[5], c[6])
        return [
            pelvis,              // 0
            c[12],               // 1  R hip
            c[14],               // 2  R knee
            c[16],               // 3  R ankle
            c[11],               // 4  L hip
            c[13],               // 5  L knee
            c[15],               // 6  L ankle
            mid(pelvis, thorax), // 7  spine
            thorax,              // 8
            c[0],                // 9  neck (nose)
            mid(c[1], c[2]),     // 10 head (mid-eye)
            c[5],                // 11 L shoulder
            c[7],                // 12 L elbow
            c[9],                // 13 L wrist
            c[6],                // 14 R shoulder
            c[8],                // 15 R elbow
            c[10],               // 16 R wrist
        ]
    }
}
