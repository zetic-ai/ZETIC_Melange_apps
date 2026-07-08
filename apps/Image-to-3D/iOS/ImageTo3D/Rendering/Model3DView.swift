import SceneKit
import SwiftUI

enum RenderMode: String, CaseIterable, Identifiable {
    case mesh = "Mesh"
    case points = "Points"
    var id: String { rawValue }
}

/// Interactive SceneKit view showing the depth relief as a textured mesh or
/// a colored point cloud (same vertices, toggled via `isHidden`).
struct Model3DView: UIViewRepresentable {
    let mesh: MeshData
    let texture: UIImage
    let contentID: Int          // bump to force a scene rebuild
    var mode: RenderMode

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.antialiasingMode = .multisampling4X
        view.autoenablesDefaultLighting = false
        view.allowsCameraControl = true
        view.defaultCameraController.interactionMode = .orbitTurntable

        // Any touch hands control to the user: stop the idle turntable sway.
        let stopSpin = UIPanGestureRecognizer(target: context.coordinator,
                                              action: #selector(Coordinator.userInteracted(_:)))
        stopSpin.cancelsTouchesInView = false
        stopSpin.delegate = context.coordinator
        view.addGestureRecognizer(stopSpin)

        rebuild(view, context: context)
        return view
    }

    func updateUIView(_ view: SCNView, context: Context) {
        if context.coordinator.contentID != contentID {
            rebuild(view, context: context)
        }
        context.coordinator.built?.meshNode.isHidden = mode != .mesh
        context.coordinator.built?.pointNode.isHidden = mode != .points
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        var contentID: Int?
        var built: SceneKitBuilder.Built?

        @objc func userInteracted(_ gesture: UIPanGestureRecognizer) {
            if gesture.state == .began {
                built?.contentNode.removeAction(forKey: "turntable")
            }
        }

        func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                               shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool {
            true
        }
    }

    private func rebuild(_ view: SCNView, context: Context) {
        let built = SceneKitBuilder.build(mesh: mesh, texture: texture)
        view.scene = built.scene
        view.pointOfView = built.cameraNode
        SceneKitBuilder.startTurntable(built.contentNode)
        context.coordinator.contentID = contentID
        context.coordinator.built = built
    }
}
