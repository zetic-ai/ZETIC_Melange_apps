import SwiftUI
import AVFoundation

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> PreviewView {
        return PreviewView(session: session)
    }
    
    func updateUIView(_ uiView: PreviewView, context: Context) {
    }
}
