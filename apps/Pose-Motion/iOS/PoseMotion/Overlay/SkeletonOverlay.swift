import SwiftUI

/// 2D COCO-17 skeleton drawn over the video, color-coded left/right/torso.
struct SkeletonOverlay: View {
    let keypoints: [Keypoint2D]
    let fit: CGRect

    // (a, b, color) — COCO indices
    private static let bones: [(Int, Int, Color)] = [
        (5, 7, Theme.leftSide), (7, 9, Theme.leftSide),        // left arm
        (6, 8, Theme.rightSide), (8, 10, Theme.rightSide),     // right arm
        (11, 13, Theme.leftSide), (13, 15, Theme.leftSide),    // left leg
        (12, 14, Theme.rightSide), (14, 16, Theme.rightSide),  // right leg
        (5, 6, Theme.torso), (11, 12, Theme.torso),            // shoulders, hips
        (5, 11, Theme.torso), (6, 12, Theme.torso),            // trunk
        (0, 5, Theme.torso.opacity(0.6)), (0, 6, Theme.torso.opacity(0.6)),  // neck
    ]

    var body: some View {
        Canvas { context, _ in
            guard keypoints.count == 17 else { return }
            let threshold = AppConfig.kptConfThreshold

            for (a, b, color) in Self.bones {
                guard keypoints[a].conf > threshold, keypoints[b].conf > threshold else { continue }
                var path = Path()
                path.move(to: point(a))
                path.addLine(to: point(b))
                context.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: 3, lineCap: .round))
            }
            for (i, kp) in keypoints.enumerated() where kp.conf > threshold {
                let p = point(i)
                let r: CGFloat = i == 0 ? 5 : 4
                context.fill(
                    Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)),
                    with: .color(i % 2 == 1 ? Theme.leftSide : (i == 0 ? Theme.torso : Theme.rightSide))
                )
            }
        }
        .allowsHitTesting(false)
    }

    private func point(_ i: Int) -> CGPoint {
        VideoFitMapper.point(CGPoint(x: keypoints[i].x, y: keypoints[i].y), in: fit)
    }
}
