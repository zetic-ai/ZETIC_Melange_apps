import AVFoundation
import SwiftUI

/// SwiftUI wrapper around the AVCaptureVideoPreviewLayer-backed view.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        PreviewView(session: session)
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}
}
