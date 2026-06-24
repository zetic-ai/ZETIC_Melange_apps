import Foundation
import ZeticMLange

struct AudioEvent: Identifiable, Equatable {
    let id = UUID()
    let index: Int
    let name: String
    let score: Float
}

/// Acoustic event detection on-device via `google/Sound Classification(YAMNET)`
/// (AudioSet, 521 classes, Apache-2.0). We surface the respiratory-relevant
/// classes (cough, breathing, wheeze, …) — the RAIsonance use case.
final class YamnetModel: ObservableObject {
    @Published var status: ModelStatus = .idle
    @Published var topEvents: [AudioEvent] = []          // overall top detections
    @Published var respiratoryEvents: [AudioEvent] = []  // filtered + sorted
    @Published var latencyMs: Double?

    private var model: ZeticMLangeModel?
    private let queue = DispatchQueue(label: "voicevitals.yamnet")
    private let classNames = YamnetLabels.load()

    /// AudioSet indices for respiratory / breath-related sounds.
    private let respiratoryIndices: [Int] = [42, 36, 37, 44, 43, 45, 39, 41, 38, 23, 54]
    // 42 Cough, 36 Breathing, 37 Wheeze, 44 Sneeze, 43 Throat clearing,
    // 45 Sniff, 39 Gasp, 41 Snort, 38 Snoring, 23 Sigh, 54 Hiccup

    var topRespiratory: AudioEvent? { respiratoryEvents.first }

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

    func analyze(_ samples: [Float]) {
        guard !status.isBusy else { return }
        status = .running

        queue.async { [weak self] in
            guard let self else { return }
            do {
                let model = try self.ensureModel()
                let fitted = MelangeKit.fit(samples, to: AppConfig.clipSamples) // 3 s window
                let input = MelangeKit.floatTensor(fitted, shape: [fitted.count]) // raw waveform
                let (outputs, ms) = try measureMs { try model.run(inputs: [input]) }

                // YAMNet emits several tensors (scores [N,521], embeddings [N,1024],
                // mel [M,64]). 521 is prime, so only the scores tensor's element count
                // is divisible by 521 — pick by that, not by shape metadata (unreliable)
                // or size (embeddings are larger and would be picked by mistake → 0%).
                guard let scoresTensor =
                        outputs.first(where: { ($0.data.count / 4) % 521 == 0 })
                        ?? outputs.first(where: { $0.shape.last == 521 })
                        ?? outputs.first else {
                    throw SimpleError("No YAMNet score tensor in outputs")
                }
                let flat = MelangeKit.floats(from: scoresTensor)
                let mean = Self.meanOverFrames(flat, classes: 521)

                let named: (Int) -> String = { idx in
                    idx < self.classNames.count ? self.classNames[idx] : "Class \(idx)"
                }
                let top = mean.enumerated()
                    .sorted { $0.element > $1.element }
                    .prefix(5)
                    .map { AudioEvent(index: $0.offset, name: named($0.offset), score: $0.element) }

                let resp = self.respiratoryIndices
                    .filter { $0 < mean.count }
                    .map { AudioEvent(index: $0, name: named($0), score: mean[$0]) }
                    .sorted { $0.score > $1.score }

                DispatchQueue.main.async {
                    self.topEvents = top
                    self.respiratoryEvents = resp
                    self.latencyMs = ms
                    self.status = .ready
                }
            } catch {
                DispatchQueue.main.async { self.status = .failed(String(describing: error)) }
            }
        }
    }

    /// YAMNet returns [frames, 521]; average the per-frame scores.
    private static func meanOverFrames(_ flat: [Float], classes: Int) -> [Float] {
        guard flat.count >= classes else { return flat }
        let frames = flat.count / classes
        guard frames > 1 else { return Array(flat.prefix(classes)) }
        var acc = [Float](repeating: 0, count: classes)
        for f in 0..<frames {
            let base = f * classes
            for c in 0..<classes { acc[c] += flat[base + c] }
        }
        let inv = 1 / Float(frames)
        return acc.map { $0 * inv }
    }

    private func ensureModel() throws -> ZeticMLangeModel {
        if let model { return model }
        let loaded = try MelangeKit.load(AppConfig.Model.yamnet) { progress in
            DispatchQueue.main.async { self.status = .downloading(progress) }
        }
        self.model = loaded
        return loaded
    }
}

/// Loads display names from the bundled `yamnet_class_map.csv` (index,mid,display_name).
enum YamnetLabels {
    static func load() -> [String] {
        guard let url = Bundle.main.url(forResource: "yamnet_class_map", withExtension: "csv"),
              let content = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        var names: [String] = []
        for line in content.split(whereSeparator: \.isNewline).dropFirst() {
            let cols = line.split(separator: ",", maxSplits: 2, omittingEmptySubsequences: false)
            guard cols.count == 3 else { continue }
            names.append(cols[2].trimmingCharacters(in: CharacterSet(charactersIn: "\" ")))
        }
        return names
    }
}
