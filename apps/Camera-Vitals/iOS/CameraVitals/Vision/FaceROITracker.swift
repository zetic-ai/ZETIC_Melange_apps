import CoreGraphics
import CoreVideo
import Vision

/// Detects the largest face with Apple Vision and returns a stable, enlarged,
/// squared ROI in pixel coordinates (top-left origin). Smooths the box and holds
/// the last good box briefly when detection drops, so cropping stays steady.
final class FaceROITracker {
    private var smoothed: CGRect?
    private var missCount = 0
    private let maxMiss = 8           // ~detections; held box before declaring "no face"
    private let smoothing: CGFloat = 0.35
    private let enlargeFactor: CGFloat = 1.5
    private let request = VNDetectFaceRectanglesRequest()   // reused across frames

    /// Returns ROI in pixel space, or nil if no face is (still) available.
    func detect(_ pixelBuffer: CVPixelBuffer) -> CGRect? {
        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return hold()
        }

        guard let face = bestFace(request.results) else { return hold() }

        // Vision: normalized, bottom-left origin → pixel, top-left origin.
        let bb = face.boundingBox
        var rect = CGRect(
            x: bb.origin.x * CGFloat(w),
            y: (1 - bb.origin.y - bb.height) * CGFloat(h),
            width: bb.width * CGFloat(w),
            height: bb.height * CGFloat(h)
        )
        rect = squareEnlarged(rect, in: CGSize(width: w, height: h))

        missCount = 0
        if let s = smoothed {
            smoothed = CGRect(
                x: s.origin.x + smoothing * (rect.origin.x - s.origin.x),
                y: s.origin.y + smoothing * (rect.origin.y - s.origin.y),
                width: s.width + smoothing * (rect.width - s.width),
                height: s.height + smoothing * (rect.height - s.height)
            )
        } else {
            smoothed = rect
        }
        return smoothed
    }

    func reset() {
        smoothed = nil
        missCount = 0
    }

    // MARK: - Helpers

    private func bestFace(_ results: [VNObservation]?) -> VNFaceObservation? {
        results?
            .compactMap { $0 as? VNFaceObservation }
            .max { $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height }
    }

    private func hold() -> CGRect? {
        missCount += 1
        if missCount > maxMiss { smoothed = nil; return nil }
        return smoothed
    }

    private func squareEnlarged(_ r: CGRect, in size: CGSize) -> CGRect {
        let side = (max(r.width, r.height) * enlargeFactor).clamped(to: 1...min(size.width, size.height))
        let cx = r.midX, cy = r.midY
        let x = (cx - side / 2).clamped(to: 0...max(0, size.width - side))
        let y = (cy - side / 2).clamped(to: 0...max(0, size.height - side))
        return CGRect(x: x, y: y, width: side, height: side)
    }
}
