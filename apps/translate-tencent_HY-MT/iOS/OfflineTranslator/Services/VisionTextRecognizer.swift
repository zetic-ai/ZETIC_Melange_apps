import Foundation
import Vision
import UIKit

/// Offline OCR via Apple's Vision framework (`VNRecognizeTextRequest`). Fully on-device. `languages`
/// are optional BCP-47 hints (empty = automatic); recognition runs off the main thread and calls
/// back on it.
enum VisionTextRecognizer {
    enum OCRError: LocalizedError {
        case invalidImage
        var errorDescription: String? { "Couldn't read the image." }
    }

    static func recognize(
        _ image: UIImage,
        languages: [String],
        completion: @escaping (Result<String, Error>) -> Void
    ) {
        guard let cgImage = image.cgImage else {
            DispatchQueue.main.async { completion(.failure(OCRError.invalidImage)) }
            return
        }

        let request = VNRecognizeTextRequest { request, error in
            if let error {
                DispatchQueue.main.async { completion(.failure(error)) }
                return
            }
            let observations = request.results as? [VNRecognizedTextObservation] ?? []
            let text = observations
                .compactMap { $0.topCandidates(1).first?.string }
                .joined(separator: "\n")
            DispatchQueue.main.async { completion(.success(text)) }
        }
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        if !languages.isEmpty {
            request.recognitionLanguages = languages
        }

        let orientation = CGImagePropertyOrientation(image.imageOrientation)
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }
    }
}

private extension CGImagePropertyOrientation {
    init(_ uiOrientation: UIImage.Orientation) {
        switch uiOrientation {
        case .up: self = .up
        case .upMirrored: self = .upMirrored
        case .down: self = .down
        case .downMirrored: self = .downMirrored
        case .left: self = .left
        case .leftMirrored: self = .leftMirrored
        case .right: self = .right
        case .rightMirrored: self = .rightMirrored
        @unknown default: self = .up
        }
    }
}
