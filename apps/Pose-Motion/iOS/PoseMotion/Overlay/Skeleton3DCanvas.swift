import SwiftUI

/// Side-view 3D skeleton: orthographic projection of the root-relative H36M joints,
/// rotated about the vertical axis by a drag-controlled azimuth.
struct Skeleton3DCanvas: View {
    let joints: [SIMD3<Float>]        // 17 H36M joints, root-relative
    @State private var azimuth: Double = 0.9   // radians; start slightly rotated
    @State private var dragStartAzimuth: Double?

    private static let bones: [(Int, Int)] = [
        (0, 1), (1, 2), (2, 3),          // right leg
        (0, 4), (4, 5), (5, 6),          // left leg
        (0, 7), (7, 8), (8, 9), (9, 10), // spine → head
        (8, 11), (11, 12), (12, 13),     // left arm
        (8, 14), (14, 15), (15, 16),     // right arm
    ]
    private static let leftJoints: Set<Int> = [4, 5, 6, 11, 12, 13]
    private static let rightJoints: Set<Int> = [1, 2, 3, 14, 15, 16]

    var body: some View {
        Canvas { context, size in
            guard joints.count == 17 else { return }
            let cosA = Float(cos(azimuth))
            let sinA = Float(sin(azimuth))

            // Rotate about Y (vertical); keep image-style y-down.
            let rotated: [(x: Float, y: Float, z: Float)] = joints.map {
                (x: $0.x * cosA + $0.z * sinA, y: $0.y, z: -$0.x * sinA + $0.z * cosA)
            }

            // Fit to canvas.
            let maxExtent = rotated.reduce(Float(1e-4)) { m, p in
                max(m, abs(p.x), abs(p.y))
            }
            let scale = CGFloat(0.42) * min(size.width, size.height) / CGFloat(maxExtent)
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            func project(_ p: (x: Float, y: Float, z: Float)) -> CGPoint {
                CGPoint(x: center.x + CGFloat(p.x) * scale, y: center.y + CGFloat(p.y) * scale)
            }

            // Depth-sorted bones, nearer drawn brighter and on top.
            let sortedBones = Self.bones.sorted {
                (rotated[$0.0].z + rotated[$0.1].z) > (rotated[$1.0].z + rotated[$1.1].z)
            }
            let zRange = rotated.map(\.z)
            let zMin = zRange.min() ?? 0
            let zMax = zRange.max() ?? 1
            func depthAlpha(_ z: Float) -> Double {
                let t = (z - zMin) / max(1e-4, zMax - zMin)
                return 0.35 + 0.65 * Double(1 - t)
            }

            for (a, b) in sortedBones {
                var path = Path()
                path.move(to: project(rotated[a]))
                path.addLine(to: project(rotated[b]))
                let color: Color =
                    Self.leftJoints.contains(b) ? Theme.leftSide :
                    Self.rightJoints.contains(b) ? Theme.rightSide : Theme.torso
                let alpha = depthAlpha((rotated[a].z + rotated[b].z) / 2)
                context.stroke(path, with: .color(color.opacity(alpha)),
                               style: StrokeStyle(lineWidth: 3, lineCap: .round))
            }
            for p in rotated {
                let pt = project(p)
                context.fill(
                    Path(ellipseIn: CGRect(x: pt.x - 2.5, y: pt.y - 2.5, width: 5, height: 5)),
                    with: .color(Theme.textPrimary.opacity(depthAlpha(p.z)))
                )
            }
        }
        .gesture(
            DragGesture(minimumDistance: 2)
                .onChanged { value in
                    if dragStartAzimuth == nil { dragStartAzimuth = azimuth }
                    azimuth = (dragStartAzimuth ?? azimuth) + Double(value.translation.width) * 0.012
                }
                .onEnded { _ in dragStartAzimuth = nil }
        )
    }
}
