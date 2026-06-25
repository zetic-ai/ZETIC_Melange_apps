import Vision
import CoreImage
import CoreVideo
import CoreGraphics

/// A face cropped to the FER model's square input, plus its location for debug overlay.
struct CroppedFace {
    /// `inputSize × inputSize` RGBA8 bytes, top-left origin (R,G,B,A per pixel).
    let rgba: [UInt8]
    let size: Int
    /// Detected face box in normalized **analysis-image** coords, top-left origin
    /// (for an optional debug overlay). Not used by inference.
    let normalizedBox: CGRect
    /// Vision's detection confidence (0…1).
    let confidence: Float
}

/// On-device face detection (Apple Vision) + square crop/resize to the FER input.
/// No second Melange model needed for detection — Vision runs on the ANE/CPU locally.
final class FaceDetector {
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private let request = VNDetectFaceRectanglesRequest()

    /// If the cropped face ever comes out upside-down on a device, flip this. The
    /// crop math is in CoreImage's bottom-left space; whether the rendered bitmap
    /// needs a vertical flip to reach top-left order is the one thing to confirm
    /// on-device (see plan risk #1). Default matches the common case.
    static var verticalFlip = true

    /// Detect the largest face in `pixelBuffer` (already upright portrait, non-mirrored
    /// from `CameraController`) and return it cropped to `AppConfig.Face.inputSize`.
    /// Returns nil when no face is found.
    func detect(_ pixelBuffer: CVPixelBuffer) -> CroppedFace? {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        do { try handler.perform([request]) } catch { return nil }
        guard let faces = request.results, !faces.isEmpty else { return nil }

        // Largest face by area.
        let face = faces.max { a, b in
            (a.boundingBox.width * a.boundingBox.height) < (b.boundingBox.width * b.boundingBox.height)
        }!

        let w = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let h = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let bb = face.boundingBox   // normalized, bottom-left origin

        // Pixel rect (bottom-left origin), expanded with margin and squared off.
        let rx = bb.minX * w, ry = bb.minY * h
        let rw = bb.width * w, rh = bb.height * h
        let cx = rx + rw / 2, cy = ry + rh / 2
        var side = max(rw, rh) * (1 + AppConfig.Face.cropMargin)
        side = min(side, w, h)
        let x = min(max(cx - side / 2, 0), w - side)
        let y = min(max(cy - side / 2, 0), h - side)
        let cropRect = CGRect(x: x, y: y, width: side, height: side)

        let size = AppConfig.Face.inputSize
        let s = CGFloat(size) / side

        var ci = CIImage(cvPixelBuffer: pixelBuffer)
            .cropped(to: cropRect)
            .transformed(by: CGAffineTransform(translationX: -x, y: -y))
            .transformed(by: CGAffineTransform(scaleX: s, y: s))
        if Self.verticalFlip {
            // Flip vertically so the rendered bitmap is top-left origin (upright face).
            ci = ci.transformed(by: CGAffineTransform(scaleX: 1, y: -1)
                .translatedBy(x: 0, y: -CGFloat(size)))
        }

        let target = CGRect(x: 0, y: 0, width: size, height: size)
        guard let cg = ciContext.createCGImage(ci, from: target) else { return nil }

        var bytes = [UInt8](repeating: 0, count: size * size * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &bytes, width: size, height: size,
                                  bitsPerComponent: 8, bytesPerRow: size * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.draw(cg, in: target)

        // Normalized box in top-left analysis-image coords for an optional overlay.
        let box = CGRect(x: bb.minX, y: 1 - bb.maxY, width: bb.width, height: bb.height)
        return CroppedFace(rgba: bytes, size: size, normalizedBox: box, confidence: face.confidence)
    }
}
