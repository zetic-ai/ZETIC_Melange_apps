import CoreGraphics
import Foundation
import ZeticMLange

enum AppConfig {
    static let personalKey = "YOUR_MLANGE_KEY"

    // MARK: Melange models
    static let detectorName = "vaibhav-zetic/YOLO26n"
    static let detectorVersion: Int? = 1
    /// RUN_AUTO's GPU candidate for this model aborts in MPSGraph ("MLIR pass
    /// manager failed") on device — pin CoreML on the Neural Engine instead.
    static let detectorTarget: Target? = .ZETIC_MLANGE_TARGET_COREML
    static let detectorAPType: APType = .NPU
    static let poseName = "realtonypark/RTMPose-s_pose_motion"
    static let poseVersion: Int? = 1            // nil = latest
    static let liftName = "realtonypark/MotionBERT-lite_pose_motion"
    static let liftVersion: Int? = 2

    // MARK: Detector (YOLO26n)
    static let detSize = 640                    // input [1,3,640,640]
    static let personClassIndex = 0
    static let ballClassIndex = 32              // COCO "sports ball"
    static let detConfThreshold: Float = 0.25
    static let iouThreshold: Float = 0.45       // raw-output fallback path only

    // MARK: Pose (RTMPose-s)
    // Input [1,3,256,192] RGB 0..1 (ImageNet normalization is baked into the graph).
    // Outputs simcc_x [1,17,384], simcc_y [1,17,512]; coords = argmax / simccSplitRatio.
    static let poseInputWidth = 192
    static let poseInputHeight = 256
    static let simccSplitRatio: Float = 2.0
    static let jointCount = 17
    /// Softmax peak probability over the SimCC bins; sharp peaks land well above this.
    static let kptConfThreshold: Float = 0.05
    /// Person box → pose crop expansion factor (box padding before 3:4 aspect fit).
    static let cropPadding: CGFloat = 1.25

    // MARK: 3D lift (MotionBERT-Lite)
    // Input [1,81,17,3]: H36M-17 joints, (x,y centered / (min(W,H)/2), conf).
    // Root-relative output.
    static let liftWindow = 81
    /// Run the lift every N processed frames (full-window re-run; the model is non-causal).
    static let liftStride = 4
    /// Frame of the output window to render. 80 (= latest) is zero-lag; 40 (= center) is
    /// more accurate but trails the video by half the window.
    static let liftReadIndex = 80

    // MARK: UI
    static let ballTrailLength = 40
    static let clipNames = ["GolfSwing", "GolfSwing2", "GolfSwing3"]
    static let clipExtension = "mp4"
}
