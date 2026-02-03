import SwiftUI
import PhotosUI

struct VideoPicker: UIViewControllerRepresentable {
    @Binding var videoURL: URL?
    @Environment(\.presentationMode) var presentationMode

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .videos
        config.selectionLimit = 1
        
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: VideoPicker

        init(parent: VideoPicker) {
            self.parent = parent
        }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            parent.presentationMode.wrappedValue.dismiss()
            
            guard let provider = results.first?.itemProvider,
                  provider.hasItemConformingToTypeIdentifier(UTType.movie.identifier) else { return }
            
            // Load file representation
            // Note: This gives a temporary URL. We might need to copy it if we want to keep it.
            // For immediate playback, temp URL works, but PHPicker often gives a copy or needs `loadFileRepresentation`.
            
            provider.loadFileRepresentation(forTypeIdentifier: UTType.movie.identifier) { url, error in
                if let error = error {
                    print("Error loading video: \(error.localizedDescription)")
                    return
                }
                
                guard let url = url else { return }
                
                // Copy to a accessible temp location as the provided URL might be inaccessible after callback returns
                let fileName = "temp_video_\(Date().timeIntervalSince1970).mov"
                let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
                
                try? FileManager.default.removeItem(at: tempURL)
                try? FileManager.default.copyItem(at: url, to: tempURL)
                
                DispatchQueue.main.async {
                    self.parent.videoURL = tempURL
                }
            }
        }
    }
}
