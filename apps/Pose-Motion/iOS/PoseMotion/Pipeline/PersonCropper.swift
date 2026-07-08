import CoreGraphics

/// Person box → pose-model crop rect: pad, then expand to the model's 3:4 (w:h)
/// aspect around the box center, clamped to the frame.
enum PersonCropper {
    static func cropRect(for personBox: CGRect, frameSize: CGSize) -> CGRect {
        let aspect = CGFloat(AppConfig.poseInputWidth) / CGFloat(AppConfig.poseInputHeight)

        let boxW = personBox.width * frameSize.width * AppConfig.cropPadding
        let boxH = personBox.height * frameSize.height * AppConfig.cropPadding
        let cx = (personBox.midX) * frameSize.width
        let cy = (personBox.midY) * frameSize.height

        // Fit to 3:4 by growing the short side.
        var w = boxW
        var h = boxH
        if w / h > aspect {
            h = w / aspect
        } else {
            w = h * aspect
        }

        var rect = CGRect(x: cx - w / 2, y: cy - h / 2, width: w, height: h)

        // Clamp: shift inside the frame first, then intersect as a last resort.
        if rect.minX < 0 { rect.origin.x = 0 }
        if rect.minY < 0 { rect.origin.y = 0 }
        if rect.maxX > frameSize.width { rect.origin.x = frameSize.width - rect.width }
        if rect.maxY > frameSize.height { rect.origin.y = frameSize.height - rect.height }
        rect = rect.intersection(CGRect(origin: .zero, size: frameSize))
        return rect
    }
}
