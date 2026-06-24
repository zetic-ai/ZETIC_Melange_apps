package ai.zetic.demo.cameravitals

/** Central configuration & tuning constants for the rPPG pipeline (mirrors iOS AppConfig). */
object AppConfig {
    // Melange model (deployed as version 2 — the low-memory 31-frame window).
    const val PERSONAL_KEY = "YOUR_MLANGE_KEY"
    const val MODEL_NAME = "realtonypark/EfficientPhys-rPPG_camera_vitals"
    const val MODEL_VERSION = 2

    // Model I/O contract.
    const val IMG_SIZE = 72
    const val CHANNELS = 3
    const val CHUNK = 30        // model output length (rPPG samples per inference)
    const val FRAMES_IN = 31    // model input frames (chunk + 1; model does diff internally)

    // Capture / cadence. stride == chunk → contiguous windows that stitch seamlessly.
    const val FPS = 30.0
    const val STRIDE = 30

    // Rolling analysis buffer of stitched rPPG samples used for the HR FFT.
    const val ANALYSIS_SAMPLES = 300       // 10 s @ 30 fps
    const val MIN_ANALYSIS_SAMPLES = 150   // 5 s before the first heart-rate reading

    // Heart-rate band.
    const val HR_BAND_LOW = 0.75   // Hz (45 bpm)
    const val HR_BAND_HIGH = 2.5   // Hz (150 bpm)
    const val MIN_BPM = 40.0
    const val MAX_BPM = 180.0

    // Quality gating.
    const val DISPLAY_QUALITY_FLOOR = 0.15
    const val LOW_LUMA_THRESHOLD = 40f

    // Guided measurement.
    const val MEASURE_DURATION_SEC = 30.0

    val frameFloatCount: Int get() = CHANNELS * IMG_SIZE * IMG_SIZE   // 15552
}
