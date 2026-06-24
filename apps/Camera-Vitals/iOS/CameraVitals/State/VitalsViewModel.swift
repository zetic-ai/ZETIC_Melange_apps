import Combine
import CoreGraphics
import Foundation

/// Observable bridge between the pipeline and the SwiftUI views. All `@Published`
/// mutations happen on the main thread (the pipeline marshals callbacks to main).
final class VitalsViewModel: ObservableObject, RPPGPipelineDelegate {
    // FSM + live readouts
    @Published var state: MeasurementState = .loadingModel(progress: 0)
    @Published var bpm: Double?
    @Published var quality: Double = 0
    @Published var waveform: [Float] = []
    @Published var latencyMs: Double = 0

    // Face / guidance
    @Published var faceFound = false
    @Published var faceBox: CGRect?
    @Published var bufferSize: CGSize = .zero
    @Published var lowLight = false
    @Published var framesFilled = 0
    @Published var warmupProgress: Double = 0   // stitch buffer filling toward first reading

    // Guided measurement
    @Published var isMeasuring = false
    @Published var measureProgress: Double = 0     // 0...1
    @Published var report: MeasurementReport?

    let camera = CameraController()
    private let model = RPPGModel()
    private var pipeline: RPPGPipeline?

    private var measureSamples: [(bpm: Double, quality: Double)] = []
    private var measureTimer: Timer?
    private var measureStart: Date?

    var isReady: Bool {
        if case .loadingModel = state { return false }
        if case .permissionDenied = state { return false }
        if case .error = state { return false }
        return true
    }

    // MARK: - Lifecycle

    func start() {
        camera.checkPermission { [weak self] granted in
            guard let self else { return }
            guard granted else { self.state = .permissionDenied; return }
            self.loadModel()
        }
    }

    private func loadModel() {
        model.load(
            onProgress: { [weak self] progress in
                guard let self else { return }
                if case .loadingModel = self.state {
                    self.state = .loadingModel(progress: progress)
                }
            },
            completion: { [weak self] result in
                guard let self else { return }
                switch result {
                case .success:
                    let pipe = RPPGPipeline(model: self.model)
                    pipe.delegate = self
                    self.pipeline = pipe
                    self.camera.onFrame = { [weak pipe] buffer in pipe?.process(buffer) }
                    self.camera.configure()
                    self.state = .warmup(framesFilled: 0)
                case .failure(let error):
                    self.state = .error(message: error.localizedDescription)
                }
            }
        )
    }

    func stop() {
        camera.stop()
        cancelMeasurement()
    }

    func retry() {
        state = .loadingModel(progress: 0)
        start()
    }

    // MARK: - Guided measurement

    func startMeasurement() {
        guard isReady, !isMeasuring else { return }
        measureSamples.removeAll()
        measureStart = Date()
        measureProgress = 0
        report = nil
        isMeasuring = true
        measureTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            self?.tickMeasurement()
        }
    }

    func cancelMeasurement() {
        measureTimer?.invalidate()
        measureTimer = nil
        isMeasuring = false
        measureProgress = 0
        measureStart = nil
    }

    private func tickMeasurement() {
        guard let start = measureStart else { return }
        let elapsed = Date().timeIntervalSince(start)
        measureProgress = min(elapsed / AppConfig.measureDuration, 1)
        if elapsed >= AppConfig.measureDuration {
            finishMeasurement()
        }
    }

    private func finishMeasurement() {
        measureTimer?.invalidate()
        measureTimer = nil
        isMeasuring = false
        measureProgress = 1

        let good = measureSamples.filter { $0.quality > AppConfig.displayQualityFloor }
        let used = good.isEmpty ? measureSamples : good
        guard !used.isEmpty else { report = nil; return }

        let bpms = used.map { $0.bpm }
        let avg = bpms.reduce(0, +) / Double(bpms.count)
        report = MeasurementReport(
            avgBPM: Int(avg.rounded()),
            minBPM: Int((bpms.min() ?? avg).rounded()),
            maxBPM: Int((bpms.max() ?? avg).rounded()),
            avgQuality: used.map { $0.quality }.reduce(0, +) / Double(used.count),
            series: bpms
        )
    }

    func dismissReport() { report = nil }

    // MARK: - RPPGPipelineDelegate

    func pipeline(_ pipeline: RPPGPipeline, didUpdateFace state: FaceState) {
        faceFound = state.faceFound
        faceBox = state.faceBox
        bufferSize = state.bufferSize
        lowLight = state.lowLight
        framesFilled = state.framesFilled
        // Warmup → live is driven by the stitch buffer (didUpdateVitals), not the frame ring,
        // since the ring now fills in ~1 s but a stable HR needs a few seconds of samples.
    }

    func pipeline(_ pipeline: RPPGPipeline, didUpdateVitals update: VitalsUpdate) {
        bpm = update.bpm
        quality = update.quality
        waveform = update.waveform
        latencyMs = update.latencyMs
        warmupProgress = update.warmupProgress

        if case .warmup = state, update.warmupProgress >= 1 {
            state = .live
        }

        if isMeasuring, let bpm = update.bpm {
            measureSamples.append((bpm: bpm, quality: update.quality))
        }
    }
}
