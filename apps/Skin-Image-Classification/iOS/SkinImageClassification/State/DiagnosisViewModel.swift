import SwiftUI
import OSLog

/// Drives the demo: owns the on-device skin classifier and runs the
/// photo → classify pipeline, publishing state for the UI.
///
/// This build is **classifier-only**. (The MedGemma LLM explanation step was dropped
/// because the 4B model's on-device load was the bottleneck; the result screen uses
/// curated per-condition guidance instead. `MedGemmaService`/`Prompts` remain in the
/// project, unused, so the LLM step can be re-enabled later.)
@MainActor
final class DiagnosisViewModel: ObservableObject {

    private let classifier = SkinClassifier()

    @Published private(set) var classifierPhase: LoadPhase = .idle
    @Published private(set) var analysis: AnalysisState = .none
    @Published private(set) var image: UIImage?
    @Published private(set) var classification: Classification?

    private var bootstrapStarted = false
    private var analysisTask: Task<Void, Never>?
    private static let log = Logger(subsystem: "ai.zetic.demo.SkinImageClassification", category: "pipeline")

    /// Capture is gated on the classifier being ready.
    var canAnalyze: Bool { classifierPhase.isReady }

    // MARK: Bootstrap

    /// Load the on-device classifier (download + compile on first run).
    func bootstrap() {
        guard !bootstrapStarted else { return }
        bootstrapStarted = true

        Task { [weak self] in
            guard let self else { return }
            self.classifierPhase = .preparing
            do {
                try await self.classifier.ensureLoaded { progress in
                    Task { @MainActor in
                        self.classifierPhase = (progress > 0 && progress < 1) ? .downloading(Double(progress)) : .preparing
                    }
                }
                self.classifierPhase = .ready
                if AppConfig.Debug.selfTestClassifierOnLaunch { self.runClassifierSelfTest() }
            } catch {
                self.classifierPhase = .failed(error.localizedDescription)
            }
        }
    }

    /// Retry a failed classifier load.
    func retryLoad() {
        if classifierPhase.errorMessage != nil { classifierPhase = .idle }
        bootstrapStarted = false
        bootstrap()
    }

    // MARK: Analysis

    func analyze(_ picked: UIImage) {
        analysisTask?.cancel()
        image = picked
        classification = nil
        analysis = .classifying

        analysisTask = Task { [weak self] in
            guard let self else { return }
            do {
                let result = try await self.classifier.classify(picked)
                if Task.isCancelled { return }
                self.classification = result
                self.analysis = .done
                MemoryProbe.log("classified: \(result.topClass.title) \(Int(result.confidence * 100))%")
            } catch is CancellationError {
                // user moved on — nothing to surface
            } catch {
                Self.log.error("classify failed: \(error.localizedDescription, privacy: .public)")
                self.analysis = .failed(error.localizedDescription)
            }
        }
    }

    /// Clear the current result and return to capture.
    func reset() {
        analysisTask?.cancel()
        analysisTask = nil
        image = nil
        classification = nil
        analysis = .none
    }

    // MARK: Diagnostics

    /// Headless self-test: drive the real `analyze()` flow on a generated image and
    /// write the outcome to Documents/selftest.log (pulled via `devicectl device copy
    /// from`). Verifies the user-facing pipeline reaches `.done` without a tap.
    private func runClassifierSelfTest() {
        Task { [weak self] in
            guard let self else { return }
            Self.debugWrite("SELFTEST start: classifier ready, calling analyze()", truncate: true)
            self.analyze(Self.makeTestImage())
            for _ in 0..<80 {   // up to ~20s
                switch self.analysis {
                case .done:
                    let c = self.classification
                    Self.debugWrite("SELFTEST DONE: \(c?.topClass.title ?? "?") \(Int((c?.confidence ?? 0) * 100))% in \(Int(c?.latencyMs ?? 0))ms")
                    return
                case .failed(let m):
                    Self.debugWrite("SELFTEST FAILED: \(m)")
                    return
                default:
                    try? await Task.sleep(nanoseconds: 250_000_000)
                }
            }
            Self.debugWrite("SELFTEST TIMEOUT: analysis=\(String(describing: self.analysis))")
        }
    }

    /// Append a diagnostic line to Documents/selftest.log on device.
    private static func debugWrite(_ s: String, truncate: Bool = false) {
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("selftest.log")
        let line = s + "\n"
        if truncate { try? line.data(using: .utf8)?.write(to: url); return }
        if let data = line.data(using: .utf8) {
            if let h = try? FileHandle(forWritingTo: url) {
                h.seekToEndOfFile(); h.write(data); try? h.close()
            } else {
                try? data.write(to: url)
            }
        }
    }

    private static func makeTestImage() -> UIImage {
        let size = CGSize(width: 224, height: 224)
        return UIGraphicsImageRenderer(size: size).image { ctx in
            UIColor(red: 0.78, green: 0.6, blue: 0.5, alpha: 1).setFill()
            ctx.fill(CGRect(origin: .zero, size: size))
            UIColor(red: 0.4, green: 0.2, blue: 0.18, alpha: 0.8).setFill()
            ctx.cgContext.fillEllipse(in: CGRect(x: 72, y: 72, width: 80, height: 80))
        }
    }
}
