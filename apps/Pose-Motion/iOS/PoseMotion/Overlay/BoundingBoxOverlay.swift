import SwiftUI

/// Person + ball boxes over the video.
struct BoundingBoxOverlay: View {
    let personBox: CGRect?    // normalized
    let ballBox: CGRect?      // normalized
    let fit: CGRect

    var body: some View {
        Canvas { context, _ in
            if let person = personBox {
                draw(context: context, rect: person, color: Theme.accent)
            }
            if let ball = ballBox {
                draw(context: context, rect: ball, color: Theme.ball)
            }
        }
        .allowsHitTesting(false)
    }

    private func draw(context: GraphicsContext, rect: CGRect, color: Color) {
        let r = VideoFitMapper.rect(rect, in: fit)
        context.stroke(
            Path(roundedRect: r, cornerRadius: 6),
            with: .color(color.opacity(0.9)),
            style: StrokeStyle(lineWidth: 2)
        )
    }
}
