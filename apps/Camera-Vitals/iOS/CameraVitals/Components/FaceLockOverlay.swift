import SwiftUI

/// Draws the locked face ROI over the camera preview, colored by signal quality.
/// Maps buffer-pixel coordinates into the aspect-fill preview rect.
struct FaceLockOverlay: View {
    let faceBox: CGRect?
    let bufferSize: CGSize
    let quality: Double
    let faceFound: Bool

    var body: some View {
        GeometryReader { geo in
            if faceFound, let box = faceBox, bufferSize.width > 0 {
                let rect = Self.mapAspectFill(box: box, buffer: bufferSize, view: geo.size)
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .stroke(Theme.quality(quality), lineWidth: 3)
                    .frame(width: rect.width, height: rect.height)
                    .position(x: rect.midX, y: rect.midY)
                    .animation(.easeOut(duration: 0.15), value: rect)
            }
        }
        .allowsHitTesting(false)
    }

    static func mapAspectFill(box: CGRect, buffer: CGSize, view: CGSize) -> CGRect {
        guard buffer.width > 0, buffer.height > 0 else { return .zero }
        let scale = max(view.width / buffer.width, view.height / buffer.height)
        let scaledW = buffer.width * scale
        let scaledH = buffer.height * scale
        let dx = (view.width - scaledW) / 2
        let dy = (view.height - scaledH) / 2
        return CGRect(
            x: box.origin.x * scale + dx,
            y: box.origin.y * scale + dy,
            width: box.width * scale,
            height: box.height * scale
        )
    }
}
