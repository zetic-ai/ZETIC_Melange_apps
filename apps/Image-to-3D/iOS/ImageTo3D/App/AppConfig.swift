import Foundation

/// Central configuration for the Image-to-3D pipeline.
enum AppConfig {
    // Melange model.
    static let personalKey = "YOUR_MLANGE_KEY"
    static let modelName = "realtonypark/Depth-Anything-V2-Small"
    static let modelVersion: Int? = nil   // nil = latest

    // Model I/O contract (fixed at export time; NPU shapes are hard-coded).
    // Input: (1, 3, 518, 518) float32, RGB in [0, 1]. ImageNet normalization
    // is baked into the exported graph. Output: (1, 518, 518) relative
    // inverse depth (larger = closer).
    static let inputSize = 518

    // Depth to 3D reconstruction (planar relief: x/y stay on a fixed reference
    // plane so the photo keeps its shape; relative depth displaces z only.
    // Mapping the full disparity range through a pinhole instead turns the
    // mesh into a deep frustum cone; background pixels blow up laterally).
    static let fovYDegrees: Float = 60          // display camera FOV
    static let planeZ: Float = 2.0              // reference plane distance
    static let reliefDepth: Float = 0.4         // z displacement as a fraction
                                                // of the photo's world width
    static let meshStride = 1                   // full 518x518 vertex grid;
                                                // quad-level silhouette steps
                                                // are invisible at this size
                                                // (mesh build ~15 ms in Release)
    static let edgeThreshold: Float = 0.07      // normalized-disparity delta above
                                                // which quads are dropped
}
