import Foundation
import ZeticMLange

/// Speech-emotion recognition on-device via `realtonypark/Wav2Vec2-Base_Emotion-Recognition`
/// (wav2vec2-large-xlsr, Apache-2.0). Input: raw 16 kHz waveform `(1, 48000)`; output: 7 logits.
///
/// This is the **voice** half of the Aiberry-style multimodal check-in — the same
/// pipeline a client would use, just with their model. Reused verbatim from VoiceVitals.
final class EmotionModel: ObservableObject {
    @Published var status: ModelStatus = .idle
    @Published var scores: [EmotionScore] = []
    @Published var latencyMs: Double?

    private var model: ZeticMLangeModel?
    private let queue = DispatchQueue(label: "aiberry.emotion")

    // Label order matches the model's config.id2label:
    // [angry, disgust, fear, happy, neutral, sad, surprise].
    let labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    var top: EmotionScore? { scores.max(by: { $0.probability < $1.probability }) }

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

    /// Analyze a captured clip and publish ranked scores. `completion` returns the
    /// final probability distribution (label order = `labels`) for the fusion engine.
    func analyze(_ samples: [Float], completion: (([Float]) -> Void)? = nil) {
        guard !status.isBusy else { return }
        status = .running

        queue.async { [weak self] in
            guard let self else { return }
            do {
                let model = try self.ensureModel()
                let n = AppConfig.clipSamples

                // 1) Trim dead air so the model's mean-pool sees mostly speech.
                let speech = AudioUtils.trimSilence(samples)
                // 2) Cover the whole utterance: average logits over overlapping windows.
                let windows = AudioUtils.windows(speech, size: n)
                    .map { MelangeKit.fitTiling($0, to: n) }

                var summed = [Float](repeating: 0, count: self.labels.count)
                var totalMs = 0.0
                for window in windows {
                    let input = MelangeKit.floatTensor(window, shape: [1, n]) // (1, 48000)
                    let (outputs, ms) = try measureMs { try model.run(inputs: [input]) }
                    totalMs += ms
                    guard let logits = outputs.first.map({ MelangeKit.floats(from: $0) }),
                          logits.count >= self.labels.count else {
                        throw SimpleError("Unexpected model output (\(outputs.first?.shape ?? []))")
                    }
                    for i in 0..<self.labels.count { summed[i] += logits[i] }
                }
                let logits = summed.map { $0 / Float(windows.count) }
                // Temperature scaling (T>1) softens over-confident logits; display only.
                let temperature: Float = 2.0
                let scaled = logits.prefix(self.labels.count).map { $0 / temperature }
                let probs = MelangeKit.softmax(Array(scaled))
                let ranked = zip(self.labels, probs)
                    .map { EmotionScore(label: $0.0, probability: $0.1) }
                    .sorted { $0.probability > $1.probability }
                DispatchQueue.main.async {
                    self.scores = ranked
                    self.latencyMs = totalMs
                    self.status = .ready
                    completion?(probs)
                }
            } catch {
                DispatchQueue.main.async {
                    self.status = .failed(String(describing: error))
                    completion?([])
                }
            }
        }
    }

    private func ensureModel() throws -> ZeticMLangeModel {
        if let model { return model }
        let loaded = try MelangeKit.load(AppConfig.Model.emotion, version: AppConfig.Model.emotionVersion) { progress in
            DispatchQueue.main.async { self.status = .downloading(progress) }
        }
        self.model = loaded
        return loaded
    }
}
