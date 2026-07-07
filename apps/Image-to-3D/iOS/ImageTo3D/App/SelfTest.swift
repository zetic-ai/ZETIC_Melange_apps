import Foundation
import SceneKit
import UIKit

/// Headless end-to-end test, triggered by launching with `--selftest`:
/// runs the full pipeline on a test image and writes the depth map, offscreen
/// 3D renders, and a stats JSON into Documents/selftest so the result can be
/// pulled off the device and inspected without touching the UI. A pushed
/// Documents/selftest_input.jpg overrides the bundled sample.
final class SelfTest {
    static let shared = SelfTest()
    private let model = DepthModel()

    func start() {
        print("[ImageTo3D] selftest starting")
        model.load(onProgress: { progress in
            print("[ImageTo3D] selftest download \(Int(progress * 100))%")
        }, completion: { [weak self] result in
            switch result {
            case .success:
                DispatchQueue.global(qos: .userInitiated).async { self?.run() }
            case .failure(let error):
                self?.finish(["status": "model load failed: \(error)"])
            }
        })
    }

    private func run() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let pushed = docs.appendingPathComponent("selftest_input.jpg")
        let url = FileManager.default.fileExists(atPath: pushed.path)
            ? pushed
            : Bundle.main.url(forResource: "sample", withExtension: "jpg")
        guard let url,
              let data = try? Data(contentsOf: url),
              let image = UIImage(data: data) else {
            finish(["status": "no test image (bundle sample.jpg / Documents/selftest_input.jpg)"])
            return
        }
        guard let input = ImagePreprocessor.prepare(image) else {
            finish(["status": "preprocess failed"])
            return
        }
        do {
            let depth = try model.infer(input.chw)
            let t0 = CFAbsoluteTimeGetCurrent()
            let mesh = DepthTo3D.build(depth: depth, texture: input.texture)
            let meshMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

            var files: [String: UIImage] = [
                "photo": input.texture,
                "depth": DepthColormap.image(from: depth) ?? UIImage(),
            ]
            DispatchQueue.main.sync {
                files["mesh_front"] = self.render(mesh: mesh, texture: input.texture, yaw: 0, points: false)
                files["mesh_left"] = self.render(mesh: mesh, texture: input.texture, yaw: -30, points: false)
                files["mesh_right"] = self.render(mesh: mesh, texture: input.texture, yaw: 30, points: false)
                files["points_right"] = self.render(mesh: mesh, texture: input.texture, yaw: 30, points: true)
            }
            write(images: files)
            finish([
                "status": "ok",
                "inferMs": Int(model.lastInferMs),
                "meshMs": Int(meshMs),
                "depthRobustRange": [depth.robustLo, depth.robustHi],
                "vertices": mesh.positions.count,
                "triangles": mesh.triangleIndices.count / 3,
            ])
        } catch {
            finish(["status": "inference failed: \(error)"])
        }
    }

    private func render(mesh: MeshData, texture: UIImage, yaw: Float, points: Bool) -> UIImage {
        let built = SceneKitBuilder.build(mesh: mesh, texture: texture)
        built.meshNode.isHidden = points
        built.pointNode.isHidden = !points
        built.contentNode.eulerAngles.y = yaw * .pi / 180
        let renderer = SCNRenderer(device: MTLCreateSystemDefaultDevice(), options: nil)
        renderer.scene = built.scene
        renderer.pointOfView = built.cameraNode
        return renderer.snapshot(atTime: 0,
                                 with: CGSize(width: 700, height: 700),
                                 antialiasingMode: .multisampling4X)
    }

    private func write(images: [String: UIImage]) {
        let dir = outputDir()
        for (name, image) in images {
            try? image.pngData()?.write(to: dir.appendingPathComponent("\(name).png"))
        }
    }

    private func finish(_ stats: [String: Any]) {
        let dir = outputDir()
        if let json = try? JSONSerialization.data(withJSONObject: stats, options: [.sortedKeys]) {
            try? json.write(to: dir.appendingPathComponent("stats.json"))
        }
        print("[ImageTo3D] selftest finished: \(stats)")
    }

    private func outputDir() -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("selftest", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}
