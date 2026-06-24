import Foundation
import CoreVideo
import ZeticMLange

/// Facial-expression recognition on-device via a ViT FER model on Melange.
/// Input: `1 × 3 × 224 × 224` raw RGB; output: 7 logits in `AppConfig.emotionLabels`
/// order (the export wrapper reorders to the canonical order — see prepare script).
///
/// Fed throttled frames from `CameraController`. Detection (Apple Vision) + crop +
/// inference all run on a private queue with a single-in-flight gate, and per-frame
/// softmax probabilities are accumulated into a session-wide running mean — robust
/// to the odd blurry/occluded frame (cf. the voice model's multi-window averaging).
final class FaceEmotionModel: ObservableObject {
    @Published var status: ModelStatus = .idle
    @Published var liveScores: [EmotionScore] = []   // running-mean distribution, ranked
    @Published var framesWithFace = 0
    @Published var latencyMs: Double?

    private var model: ZeticMLangeModel?
    private let queue = DispatchQueue(label: "aiberry.face")
    private let detector = FaceDetector()
    // Single-in-flight gate. Checked synchronously in `ingest` so a new frame is
    // dropped *before* it's captured into a queued closure — bounds work to one
    // inference at a time and prevents camera buffers piling up in memory.
    private let gate = NSLock()
    private var busy = false

    private let labels = AppConfig.emotionLabels        // canonical order (output/accumulator)
    private let modelLabels = AppConfig.faceModelLabels // hosted model's native logit order
    /// modelLabels[i] → canonical index, so we can read native logits and accumulate
    /// in canonical order (keeps fusion/styling model-agnostic).
    private let toCanonical: [Int]
    private var summed: [Float]                          // running sum of per-frame softmax
    private var frames = 0

    var liveTop: EmotionScore? { liveScores.max(by: { $0.probability < $1.probability }) }

    init() {
        summed = [Float](repeating: 0, count: AppConfig.emotionLabels.count)
        toCanonical = AppConfig.faceModelLabels.map { native in
            let canonical = AppConfig.faceLabelToCanonical[native] ?? native
            return AppConfig.emotionLabels.firstIndex(of: canonical) ?? 0
        }
    }

    /// Download + compile the model ahead of time (called at launch).
    func preload() {
        guard model == nil, !status.isBusy else { return }
        status = .downloading(0)
        queue.async { [weak self] in
            guard let self else { return }
            do { _ = try self.ensureModel(); DispatchQueue.main.async { self.status = .idle } }
            catch { DispatchQueue.main.async { self.status = .failed(String(describing: error)) } }
        }
    }

    /// Begin a fresh accumulation for a new check-in.
    func reset() {
        queue.async { [weak self] in
            guard let self else { return }
            self.summed = [Float](repeating: 0, count: self.labels.count)
            self.frames = 0
            DispatchQueue.main.async {
                self.liveScores = []
                self.framesWithFace = 0
                self.latencyMs = nil
            }
        }
    }

    /// Ingest one camera frame. Dropped if a previous frame is still being processed
    /// or if the model isn't ready. Frames with no detected face are skipped.
    func ingest(_ pixelBuffer: CVPixelBuffer) {
        // Drop now (without capturing the buffer) if an inference is already running.
        gate.lock()
        if busy { gate.unlock(); return }
        busy = true
        gate.unlock()

        queue.async { [weak self] in
            guard let self else { return }
            defer { self.gate.lock(); self.busy = false; self.gate.unlock() }
            guard let model = self.model else { return }

            guard let face = self.detector.detect(pixelBuffer) else { return }
            let values = FacePixelTensor.bgrMeanSubtracted(face.rgba, size: face.size)
            let size = face.size
            let input = MelangeKit.floatTensor(values, shape: [1, 3, size, size])
            do {
                let (outputs, ms) = try measureMs { try model.run(inputs: [input]) }
                guard let logits = outputs.first.map({ MelangeKit.floats(from: $0) }),
                      logits.count >= self.modelLabels.count else { return }
                // Softmax over the model's native order, then accumulate in canonical order.
                let probs = MelangeKit.softmax(Array(logits.prefix(self.modelLabels.count)))
                // Weight each frame by detection confidence so a marginal detection
                // counts less toward the running mean.
                let w = max(0.2, face.confidence)
                for i in 0..<self.modelLabels.count { self.summed[self.toCanonical[i]] += probs[i] * w }
                self.frames += 1
                let mean = self.summed.map { $0 / Float(self.frames) }
                let ranked = zip(self.labels, mean)
                    .map { EmotionScore(label: $0.0, probability: $0.1) }
                    .sorted { $0.probability > $1.probability }
                DispatchQueue.main.async {
                    self.liveScores = ranked
                    self.framesWithFace = self.frames
                    self.latencyMs = ms
                }
            } catch {
                DispatchQueue.main.async { self.status = .failed(String(describing: error)) }
            }
        }
    }

    /// The session-wide face distribution (probabilities by `AppConfig.emotionLabels`),
    /// plus how many good frames it was built from. Nil if no face was ever seen.
    func finalize(_ completion: @escaping (_ distribution: [Float], _ framesWithFace: Int) -> Void) {
        queue.async { [weak self] in
            guard let self else { return }
            let f = self.frames
            guard f > 0 else { DispatchQueue.main.async { completion([], 0) }; return }
            // Renormalize the (confidence-weighted) running sum to a probability vector.
            let total = self.summed.reduce(0, +)
            let dist = total > 0 ? self.summed.map { $0 / total } : self.summed
            DispatchQueue.main.async { completion(dist, f) }
        }
    }

    private func ensureModel() throws -> ZeticMLangeModel {
        if let model { return model }
        let loaded = try MelangeKit.load(AppConfig.Model.face, version: AppConfig.Model.faceVersion) { progress in
            DispatchQueue.main.async { self.status = .downloading(progress) }
        }
        self.model = loaded
        return loaded
    }
}
