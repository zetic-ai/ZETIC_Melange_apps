import UIKit

/// Turns a photo into the model's input buffer.
/// Center-crops to a square (no stretching — a warped photo warps the 3D
/// result), resizes to 518×518, and emits CHW planar RGB floats in [0, 1].
/// ImageNet normalization is baked into the exported model graph.
enum ImagePreprocessor {
    struct Output {
        let chw: [Float]        // [3 * 518 * 518], planar R,G,B
        let texture: UIImage    // the cropped 518×518 image (3D mesh texture + photo pane)
    }

    static func prepare(_ image: UIImage) -> Output? {
        let side = CGFloat(AppConfig.inputSize)
        let size = CGSize(width: side, height: side)

        // Draw with aspect-fill so the photo is center-cropped, not stretched.
        // UIGraphicsImageRenderer also bakes in the EXIF orientation.
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let cropped = UIGraphicsImageRenderer(size: size, format: format).image { _ in
            let scale = max(side / image.size.width, side / image.size.height)
            let drawSize = CGSize(width: image.size.width * scale,
                                  height: image.size.height * scale)
            let origin = CGPoint(x: (side - drawSize.width) / 2,
                                 y: (side - drawSize.height) / 2)
            image.draw(in: CGRect(origin: origin, size: drawSize))
        }
        guard let cgImage = cropped.cgImage else { return nil }

        let width = AppConfig.inputSize
        let height = AppConfig.inputSize
        let totalPixels = width * height

        // Force RGBA layout (R G B A per pixel).
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue)
        var pixels = [UInt8](repeating: 0, count: totalPixels * 4)
        let drawn: Bool = pixels.withUnsafeMutableBufferPointer { ptr in
            guard let context = CGContext(data: ptr.baseAddress,
                                          width: width,
                                          height: height,
                                          bitsPerComponent: 8,
                                          bytesPerRow: width * 4,
                                          space: colorSpace,
                                          bitmapInfo: bitmapInfo.rawValue) else { return false }
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { return nil }

        // Packed RGBA [H, W, 4] → planar RGB [3, H, W] in [0, 1].
        var chw = [Float](repeating: 0, count: 3 * totalPixels)
        for i in 0..<totalPixels {
            let offset = i * 4
            chw[i] = Float(pixels[offset]) / 255.0
            chw[totalPixels + i] = Float(pixels[offset + 1]) / 255.0
            chw[2 * totalPixels + i] = Float(pixels[offset + 2]) / 255.0
        }

        return Output(chw: chw, texture: cropped)
    }
}
