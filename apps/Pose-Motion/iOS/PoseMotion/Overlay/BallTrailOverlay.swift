import SwiftUI

/// Fading trail through the last N ball centers, with a bright head dot.
struct BallTrailOverlay: View {
    let trail: [CGPoint]      // normalized frame coords, oldest first
    let fit: CGRect

    var body: some View {
        Canvas { context, _ in
            guard trail.count > 1 else { return }
            let points = trail.map { VideoFitMapper.point($0, in: fit) }

            for i in 1..<points.count {
                let alpha = Double(i) / Double(points.count)
                var segment = Path()
                segment.move(to: points[i - 1])
                segment.addLine(to: points[i])
                context.stroke(
                    segment,
                    with: .color(Theme.ball.opacity(0.15 + 0.75 * alpha)),
                    style: StrokeStyle(lineWidth: 2 + 2 * alpha, lineCap: .round)
                )
            }
            if let head = points.last {
                context.fill(
                    Path(ellipseIn: CGRect(x: head.x - 6, y: head.y - 6, width: 12, height: 12)),
                    with: .color(Theme.ball)
                )
                context.stroke(
                    Path(ellipseIn: CGRect(x: head.x - 9, y: head.y - 9, width: 18, height: 18)),
                    with: .color(Theme.ball.opacity(0.5)),
                    lineWidth: 2
                )
            }
        }
        .allowsHitTesting(false)
    }
}
