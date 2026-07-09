import UIKit

/// Turbo colormap (Google) via the standard degree-5 polynomial approximation.
/// Warm = near, cool = far.
enum DepthColormap {
    static func turbo(_ t: Float) -> (r: Float, g: Float, b: Float) {
        let x = min(max(t, 0), 1)
        let r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
        let g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
        let b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
        return (min(max(r, 0), 1), min(max(g, 0), 1), min(max(b, 0), 1))
    }

    /// Renders a depth map to a colormapped UIImage (1 = closest = warm).
    static func image(from depth: DepthMap) -> UIImage? {
        let width = depth.width
        let height = depth.height
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            let (r, g, b) = turbo(depth.normalized(i))
            let offset = i * 4
            pixels[offset] = UInt8(r * 255)
            pixels[offset + 1] = UInt8(g * 255)
            pixels[offset + 2] = UInt8(b * 255)
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue)
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let cgImage = CGImage(width: width,
                                    height: height,
                                    bitsPerComponent: 8,
                                    bitsPerPixel: 32,
                                    bytesPerRow: width * 4,
                                    space: colorSpace,
                                    bitmapInfo: bitmapInfo,
                                    provider: provider,
                                    decode: nil,
                                    shouldInterpolate: false,
                                    intent: .defaultIntent) else { return nil }
        return UIImage(cgImage: cgImage)
    }
}
