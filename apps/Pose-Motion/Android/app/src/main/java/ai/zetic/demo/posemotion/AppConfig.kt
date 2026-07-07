package ai.zetic.demo.posemotion

object AppConfig {
    const val PERSONAL_KEY = "YOUR_MLANGE_KEY"

    // Melange models (same deployments as the iOS app)
    const val DETECTOR_NAME = "vaibhav-zetic/YOLO26n"
    const val DETECTOR_VERSION = 1
    const val POSE_NAME = "realtonypark/RTMPose-s_pose_motion"
    const val POSE_VERSION = 1
    const val LIFT_NAME = "realtonypark/MotionBERT-lite_pose_motion"
    const val LIFT_VERSION = 2

    // Detector (YOLO26n), input [1,3,640,640] RGB 0..1
    const val DET_SIZE = 640
    const val PERSON_CLASS_INDEX = 0
    const val BALL_CLASS_INDEX = 32          // COCO "sports ball"
    const val DET_CONF_THRESHOLD = 0.25f
    const val IOU_THRESHOLD = 0.45f          // raw-output fallback path only

    // Pose (RTMPose-s), input [1,3,256,192] RGB 0..1 (ImageNet norm baked into the graph).
    // Outputs simcc_x [1,17,384] and simcc_y [1,17,512]; coords = argmax / SIMCC_SPLIT_RATIO.
    const val POSE_INPUT_WIDTH = 192
    const val POSE_INPUT_HEIGHT = 256
    const val SIMCC_SPLIT_RATIO = 2.0f
    const val JOINT_COUNT = 17
    /** Softmax peak probability over the SimCC bins; sharp peaks land well above this. */
    const val KPT_CONF_THRESHOLD = 0.05f
    /** Person box → pose crop expansion factor (box padding before 3:4 aspect fit). */
    const val CROP_PADDING = 1.25f

    // 3D lift (MotionBERT-Lite), input [1,81,17,3]: H36M-17 joints,
    // (x,y centered / (min(W,H)/2), conf). Root-relative output.
    const val LIFT_WINDOW = 81
    /** Run the lift every N processed frames (full-window re-run; the model is non-causal). */
    const val LIFT_STRIDE = 4
    /** Output window frame to render. 80 (= latest) is zero-lag; 40 (= center) is more accurate. */
    const val LIFT_READ_INDEX = 80

    // UI
    const val BALL_TRAIL_LENGTH = 40
    val CLIP_FILES = listOf("GolfSwing.mp4", "GolfSwing2.mp4", "GolfSwing3.mp4")
}
