import SwiftUI
import UIKit

/// Orchestrates pick → preprocess → depth inference → colormap + relief mesh.
final class ImageTo3DViewModel: ObservableObject {
    enum Phase: Equatable {
        case loadingModel(Float)   // download/compile progress
        case idle                  // model ready, waiting for a photo
        case processing(String)    // stage label
        case ready
        case error(String)
    }

    struct Latency {
        var modelLoadMs: Double = 0
        var depthMs: Double = 0
        var meshMs: Double = 0
    }

    @Published var phase: Phase = .loadingModel(0)
    @Published var latency = Latency()
    @Published var photo: UIImage?
    @Published var depthImage: UIImage?
    @Published var mesh: MeshData?
    @Published var texture: UIImage?
    @Published var contentID = 0
    @Published var renderMode: RenderMode = .mesh

    private let model = DepthModel()

    func loadModel() {
        guard case .loadingModel = phase else { return }
        let t0 = CFAbsoluteTimeGetCurrent()
        model.load(onProgress: { [weak self] progress in
            self?.phase = .loadingModel(progress)
        }, completion: { [weak self] result in
            guard let self else { return }
            self.latency.modelLoadMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            switch result {
            case .success:
                self.phase = .idle
                if let pending = self.photo, self.mesh == nil {
                    self.process(pending)
                }
            case .failure(let error):
                self.phase = .error("Model load failed: \(error.localizedDescription)")
            }
        })
    }

    func process(_ image: UIImage) {
        photo = image
        if case .loadingModel = phase { return }   // will run once loaded
        phase = .processing("Preprocessing…")

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            guard let input = ImagePreprocessor.prepare(image) else {
                DispatchQueue.main.async { self.phase = .error("Could not read the photo") }
                return
            }
            DispatchQueue.main.async { self.phase = .processing("Running depth model…") }

            do {
                let depth = try self.model.infer(input.chw)
                DispatchQueue.main.async { self.phase = .processing("Building 3D mesh…") }

                let t1 = CFAbsoluteTimeGetCurrent()
                let mesh = DepthTo3D.build(depth: depth, texture: input.texture)
                let meshMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000
                let depthImage = DepthColormap.image(from: depth)

                DispatchQueue.main.async {
                    self.photo = input.texture
                    self.depthImage = depthImage
                    self.mesh = mesh
                    self.texture = input.texture
                    self.contentID += 1
                    self.latency.depthMs = self.model.lastInferMs
                    self.latency.meshMs = meshMs
                    self.phase = .ready
                }
            } catch {
                DispatchQueue.main.async {
                    self.phase = .error("Inference failed: \(error.localizedDescription)")
                }
            }
        }
    }
}
