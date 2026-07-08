package ai.zetic.demo.imageto3d

/** Central configuration - mirrors iOS/ImageTo3D/App/AppConfig.swift. */
object AppConfig {
    // Melange model.
    const val PERSONAL_KEY = "YOUR_MLANGE_KEY"
    const val MODEL_NAME = "realtonypark/Depth-Anything-V2-Small"

    // Model I/O contract (fixed at export time; NPU shapes are hard-coded).
    // Input: (1, 3, 518, 518) float32, RGB in [0, 1]. ImageNet normalization
    // is baked into the exported graph. Output: (1, 518, 518) relative
    // inverse depth (larger = closer).
    const val INPUT_SIZE = 518

    // Depth to 3D reconstruction (planar relief: x/y stay on a fixed reference
    // plane so the photo keeps its shape; relative depth displaces z only).
    const val FOV_Y_DEGREES = 60f
    const val PLANE_Z = 2.0f
    const val RELIEF_DEPTH = 0.4f       // z displacement as a fraction of width
    const val MESH_STRIDE = 1           // full 518×518 vertex grid
    const val EDGE_THRESHOLD = 0.07f    // raw-disparity delta above which quads drop
}
