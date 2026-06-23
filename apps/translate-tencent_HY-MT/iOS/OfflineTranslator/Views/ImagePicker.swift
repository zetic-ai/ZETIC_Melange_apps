import SwiftUI
import UIKit
import PhotosUI

/// Presents a camera capture or photo-library pick and returns a `UIImage`. Camera uses
/// `UIImagePickerController`; library uses `PHPickerViewController` (no photo-library permission
/// needed for single picks).
struct ImagePicker: UIViewControllerRepresentable {
    enum Source: Identifiable {
        case camera, library
        var id: Int { self == .camera ? 0 : 1 }
    }

    let source: Source
    let onImage: (UIImage?) -> Void
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIViewController {
        switch source {
        case .camera:
            let picker = UIImagePickerController()
            picker.sourceType = .camera
            picker.delegate = context.coordinator
            return picker
        case .library:
            var config = PHPickerConfiguration()
            config.filter = .images
            config.selectionLimit = 1
            let picker = PHPickerViewController(configuration: config)
            picker.delegate = context.coordinator
            return picker
        }
    }

    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate, PHPickerViewControllerDelegate {
        private let parent: ImagePicker
        init(_ parent: ImagePicker) { self.parent = parent }

        // Camera
        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
        ) {
            parent.onImage(info[.originalImage] as? UIImage)
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.onImage(nil)
            parent.dismiss()
        }

        // Library
        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            guard let provider = results.first?.itemProvider,
                  provider.canLoadObject(ofClass: UIImage.self) else {
                parent.onImage(nil)
                parent.dismiss()
                return
            }
            provider.loadObject(ofClass: UIImage.self) { [parent] object, _ in
                DispatchQueue.main.async {
                    parent.onImage(object as? UIImage)
                    parent.dismiss()
                }
            }
        }
    }
}
