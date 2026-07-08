import Combine
import CoreGraphics
import Foundation
import SwiftUI
import ZeticMLange

struct ModelLoadState: Identifiable {
    let id: String            // display name
    var progress: Float = 0
    var loaded = false
    var error: String?
    var optional = false
}

@MainActor
final class DemoViewModel: ObservableObject {
    enum Phase {
        case loading
        case missingClip
        case running
    }

    @Published var phase: Phase = .loading
    @Published var loadStates: [ModelLoadState] = [
        ModelLoadState(id: "YOLO26n · detector"),
        ModelLoadState(id: "RTMPose-s · 2D pose"),
        ModelLoadState(id: "MotionBERT-Lite · 3D lift", optional: true),
    ]
    @Published var mode: ClipFrameSource.Mode = .benchmark
    @Published var show3D = true
    @Published var availableClips: [String] = []
    @Published var selectedClip = AppConfig.clipNames[0]

    // Latest frame state
    @Published var frameImage: CGImage?
    @Published var frameSize: CGSize = .zero
    @Published var personBox: CGRect?
    @Published var ballBox: CGRect?
    @Published var keypoints: [Keypoint2D]?
    @Published var pose3D: [SIMD3<Float>]?
    @Published var ballTrail: [CGPoint] = []
    @Published var stats = BenchmarkSnapshot()

    private let pipeline = PosePipeline()
    private var source: ClipFrameSource?

    var liftAvailable: Bool { pipeline.lift.isLoaded }

    var allRequiredLoaded: Bool {
        loadStates.allSatisfy { $0.loaded || ($0.optional && $0.error != nil) }
    }

    /// Bundled (blue-folder Media/ or bundle root), then Documents for pushing a clip
    /// via Finder/Files without rebuilding.
    /// ("Media", not "Resources" — a top-level Resources/ dir breaks iOS codesign.)
    private func clipURL(for name: String) -> URL? {
        if let url = Bundle.main.url(forResource: name,
                                     withExtension: AppConfig.clipExtension,
                                     subdirectory: "Media") { return url }
        if let url = Bundle.main.url(forResource: name,
                                     withExtension: AppConfig.clipExtension) { return url }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        if let url = docs?.appendingPathComponent("\(name).\(AppConfig.clipExtension)"),
           FileManager.default.fileExists(atPath: url.path) { return url }
        return nil
    }

    /// The configured clips that actually resolve on this device, in config order.
    private func refreshAvailableClips() {
        availableClips = AppConfig.clipNames.filter { clipURL(for: $0) != nil }
        if !availableClips.contains(selectedClip), let first = availableClips.first {
            selectedClip = first
        }
    }

    var clipURL: URL? { clipURL(for: selectedClip) }

    init() {
        pipeline.onResult = { [weak self] result, snapshot in
            self?.apply(result, snapshot)
        }
        loadModels()
    }

    func loadModels() {
        load(index: 0, model: pipeline.detector,
             name: AppConfig.detectorName, version: AppConfig.detectorVersion,
             target: AppConfig.detectorTarget, apType: AppConfig.detectorAPType)
        load(index: 1, model: pipeline.pose,
             name: AppConfig.poseName, version: AppConfig.poseVersion)
        load(index: 2, model: pipeline.lift,
             name: AppConfig.liftName, version: AppConfig.liftVersion)
    }

    func retry(index: Int) {
        loadStates[index].error = nil
        loadStates[index].progress = 0
        switch index {
        case 0: load(index: 0, model: pipeline.detector,
                     name: AppConfig.detectorName, version: AppConfig.detectorVersion,
                     target: AppConfig.detectorTarget, apType: AppConfig.detectorAPType)
        case 1: load(index: 1, model: pipeline.pose,
                     name: AppConfig.poseName, version: AppConfig.poseVersion)
        default: load(index: 2, model: pipeline.lift,
                      name: AppConfig.liftName, version: AppConfig.liftVersion)
        }
    }

    func startIfReady() {
        guard allRequiredLoaded, phase == .loading else { return }
        refreshAvailableClips()
        guard let url = clipURL else {
            phase = .missingClip
            return
        }
        phase = .running
        start(url: url)
    }

    /// Re-checks for a clip (missing-clip screen "try again").
    func recheckClip() {
        guard phase == .missingClip else { return }
        refreshAvailableClips()
        if let url = clipURL {
            phase = .running
            start(url: url)
        }
    }

    func setClip(_ name: String) {
        guard name != selectedClip else { return }
        selectedClip = name
        guard phase == .running, let url = clipURL else { return }
        source?.stop()
        pipeline.resetSession()
        ballTrail.removeAll()
        frameImage = nil
        start(url: url)
    }

    func setMode(_ newMode: ClipFrameSource.Mode) {
        guard newMode != mode else { return }
        mode = newMode
        guard phase == .running, let url = clipURL else { return }
        source?.stop()
        pipeline.resetSession()
        ballTrail.removeAll()
        start(url: url)
    }

    private func start(url: URL) {
        let s = ClipFrameSource(url: url, mode: mode, queue: pipeline.queue)
        s.onFrame = { [weak self] pixelBuffer, time in
            self?.pipeline.process(pixelBuffer, at: time)
        }
        source = s
        s.start()
    }

    private func load(index: Int, model: MelangeModel, name: String, version: Int?,
                      target: Target? = nil, apType: APType = .NA) {
        model.load(
            name: name, version: version, target: target, apType: apType,
            onProgress: { [weak self] p in self?.loadStates[index].progress = p },
            completion: { [weak self] result in
                guard let self else { return }
                switch result {
                case .success:
                    self.loadStates[index].loaded = true
                case .failure(let error):
                    self.loadStates[index].error = error.localizedDescription
                }
                self.startIfReady()
            }
        )
    }

    private func apply(_ result: FrameResult, _ snapshot: BenchmarkSnapshot) {
        frameImage = result.image
        frameSize = result.frameSize
        personBox = result.personBox
        ballBox = result.ballBox
        keypoints = result.keypoints
        pose3D = result.pose3D
        stats = snapshot

        if let ball = result.ballBox {
            ballTrail.append(CGPoint(x: ball.midX, y: ball.midY))
            if ballTrail.count > AppConfig.ballTrailLength {
                ballTrail.removeFirst(ballTrail.count - AppConfig.ballTrailLength)
            }
        }
    }
}
