import CoreGraphics

/// Aspect-fit math shared by every overlay: where the video frame lands inside the view.
enum VideoFitMapper {
    static func fitRect(content: CGSize, in container: CGSize) -> CGRect {
        guard content.width > 0, content.height > 0,
              container.width > 0, container.height > 0 else { return .zero }
        let scale = min(container.width / content.width, container.height / content.height)
        let size = CGSize(width: content.width * scale, height: content.height * scale)
        return CGRect(
            x: (container.width - size.width) / 2,
            y: (container.height - size.height) / 2,
            width: size.width,
            height: size.height
        )
    }

    /// Normalized (0..1) frame point → view point.
    static func point(_ p: CGPoint, in fit: CGRect) -> CGPoint {
        CGPoint(x: fit.minX + p.x * fit.width, y: fit.minY + p.y * fit.height)
    }

    /// Normalized (0..1) frame rect → view rect.
    static func rect(_ r: CGRect, in fit: CGRect) -> CGRect {
        CGRect(
            x: fit.minX + r.minX * fit.width,
            y: fit.minY + r.minY * fit.height,
            width: r.width * fit.width,
            height: r.height * fit.height
        )
    }
}
