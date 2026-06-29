import UIKit
import ZeticMLange

/// Turns a user photo into the exact input tensor the on-device ViT expects.
///
/// Steps: normalize EXIF orientation → optional center-crop to square → scale to
/// 224×224 RGBA8 via CoreGraphics → deinterleave into a planar/interleaved Float
/// buffer with the configured channel order + normalization.
///
/// **The layout, channel order, and normalization are all read from
/// `AppConfig.Preprocess`** because the Melange-converted graph may not match the
/// PyTorch contract until verified on-device (see README "Validate the classifier").
enum ImagePreprocessor {

    /// Produce the classifier input tensor for `image`, or nil if rendering fails.
    static func tensor(from image: UIImage) -> Tensor? {
        let size = AppConfig.Preprocess.inputSize
        guard let pixels = rgbaPixels(from: image, side: size) else { return nil }

        let n = size * size
        var out = [Float](repeating: 0, count: 3 * n)

        // Per-channel source offsets within each RGBA pixel.
        let (c0, c1, c2): (Int, Int, Int)
        switch AppConfig.Preprocess.channelOrder {
        case .rgb: (c0, c1, c2) = (0, 1, 2)   // plane0=R, plane1=G, plane2=B
        case .bgr: (c0, c1, c2) = (2, 1, 0)   // plane0=B, plane1=G, plane2=R
        }

        let normalize: (UInt8) -> Float
        switch AppConfig.Preprocess.normalize {
        case .signed1: normalize = { Float($0) / 127.5 - 1.0 }   // [-1, 1]
        case .unit:    normalize = { Float($0) / 255.0 }          // [0, 1]
        }

        for i in 0..<n {
            let p = i * 4
            let v0 = normalize(pixels[p + c0])
            let v1 = normalize(pixels[p + c1])
            let v2 = normalize(pixels[p + c2])
            switch AppConfig.Preprocess.layout {
            case .nchw:                       // [1,3,H,W] — channel planes
                out[i] = v0
                out[n + i] = v1
                out[2 * n + i] = v2
            case .nhwc:                       // [1,H,W,3] — interleaved
                out[i * 3] = v0
                out[i * 3 + 1] = v1
                out[i * 3 + 2] = v2
            }
        }

        let shape = AppConfig.Preprocess.layout == .nchw
            ? [1, 3, size, size]
            : [1, size, size, 3]
        return MelangeKit.floatTensor(out, shape: shape)
    }

    /// Draw `image` (orientation-corrected, optionally center-cropped square) into a
    /// `side`×`side` RGBA8 buffer. Byte order is R,G,B,A.
    private static func rgbaPixels(from image: UIImage, side: Int) -> [UInt8]? {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        var buffer = [UInt8](repeating: 0, count: side * side * 4)

        let ok: Bool = buffer.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(
                data: raw.baseAddress,
                width: side, height: side,
                bitsPerComponent: 8, bytesPerRow: side * 4,
                space: colorSpace, bitmapInfo: bitmapInfo
            ) else { return false }
            ctx.interpolationQuality = .high

            // Drawing the UIImage (not the raw CGImage) applies EXIF orientation, so a
            // sideways library photo doesn't wreck the prediction.
            UIGraphicsPushContext(ctx)
            defer { UIGraphicsPopContext() }
            let rect = CGRect(x: 0, y: 0, width: side, height: side)
            if AppConfig.Preprocess.centerCropSquare {
                drawCenterCropped(image, in: rect)
            } else {
                image.draw(in: rect)
            }
            return true
        }
        return ok ? buffer : nil
    }

    /// Aspect-fill: scale so the shorter side fills `rect`, centering the overflow —
    /// equivalent to a center crop to square, matching ViT's training transform.
    private static func drawCenterCropped(_ image: UIImage, in rect: CGRect) {
        let imgSize = image.size
        guard imgSize.width > 0, imgSize.height > 0 else { image.draw(in: rect); return }
        let scale = max(rect.width / imgSize.width, rect.height / imgSize.height)
        let drawSize = CGSize(width: imgSize.width * scale, height: imgSize.height * scale)
        let origin = CGPoint(x: rect.midX - drawSize.width / 2, y: rect.midY - drawSize.height / 2)
        image.draw(in: CGRect(origin: origin, size: drawSize))
    }
}
