import SwiftUI
import AVFoundation

/// SwiftUI wrapper over the `CameraController`'s preview layer — the mirrored
/// front-camera self-view shown as a small PiP during the check-in.
struct CameraPreviewView: UIViewRepresentable {
    let controller: CameraController

    func makeUIView(context: Context) -> PreviewContainer {
        let v = PreviewContainer()
        v.backgroundColor = .black
        v.layer.addSublayer(controller.previewLayer)
        return v
    }

    func updateUIView(_ uiView: PreviewContainer, context: Context) {}

    /// Hosts the preview layer and keeps it sized to the view's bounds.
    final class PreviewContainer: UIView {
        override func layoutSubviews() {
            super.layoutSubviews()
            layer.sublayers?.forEach { $0.frame = bounds }
        }
    }
}
