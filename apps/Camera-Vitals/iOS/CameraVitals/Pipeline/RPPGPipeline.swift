import CoreGraphics
import CoreVideo
import Foundation

/// Buffer/ROI sizes reported alongside results, so overlays can map ROI → preview.
struct FaceState {
    let faceFound: Bool
    let faceBox: CGRect?          // pixel coords in the (portrait) buffer
    let bufferSize: CGSize
    let lowLight: Bool
    let framesFilled: Int
    let capacity: Int
}

struct VitalsUpdate {
    let bpm: Double?
    let quality: Double
    let waveform: [Float]
    let latencyMs: Double
    let warmupProgress: Double   // 0...1 — stitch buffer filling toward the first reading
}

protocol RPPGPipelineDelegate: AnyObject {
    func pipeline(_ pipeline: RPPGPipeline, didUpdateFace state: FaceState)
    func pipeline(_ pipeline: RPPGPipeline, didUpdateVitals update: VitalsUpdate)
}

/// Orchestrates the temporal pipeline: face crop → ring buffer → sliding-window inference → HR.
/// `process(_:)` is called on the capture queue; inference runs on its own queue with a
/// single-flight `busy` gate (drop windows under back-pressure, never queue them).
final class RPPGPipeline {
    weak var delegate: RPPGPipelineDelegate?

    private let model: RPPGModel
    private let tracker = FaceROITracker()
    private let ring = FrameRingBuffer(capacity: AppConfig.framesIn, frameLen: AppConfig.frameFloatCount)
    private let inferenceQueue = DispatchQueue(label: "rppg.inference", qos: .userInitiated)

    private var frameCounter = 0
    private var detectCounter = 0
    private var lastBox: CGRect?
    private var busy = false
    private let busyLock = NSLock()

    // Stitched rPPG samples + display waveform + BPM smoothing — touched only on the inference queue.
    private var rawWaveform = [Float]()       // raw model outputs, stitched (rolling)
    private var displayWaveform = [Float]()   // band-passed, for the chart
    private var bpmFilter = MedianEMA(size: 5, alpha: 0.3)
    private var inferenceCount = 0

    init(model: RPPGModel) { self.model = model }

    func reset() {
        ring.reset()
        tracker.reset()
        frameCounter = 0
        detectCounter = 0
        lastBox = nil
        setBusy(false)
        inferenceQueue.async {
            self.rawWaveform.removeAll()
            self.displayWaveform.removeAll()
            self.bpmFilter.reset()
        }
    }

    /// Called on the capture queue for every frame. The whole body runs inside an
    /// autoreleasepool: this queue is never idle, so without an explicit pool the
    /// autoreleased temporaries from Vision + Core Video would accumulate for the
    /// entire session and eventually get the app jetsam-killed.
    func process(_ pixelBuffer: CVPixelBuffer) {
        autoreleasepool {
            let bufferSize = CGSize(
                width: CVPixelBufferGetWidth(pixelBuffer),
                height: CVPixelBufferGetHeight(pixelBuffer)
            )

            // Detect + publish face state every 3rd frame (~10 Hz) for stability and CPU
            // headroom — but still crop/buffer EVERY frame so the model gets full 30 fps.
            detectCounter &+= 1
            let doDetect = detectCounter % 3 == 0
            if doDetect { lastBox = tracker.detect(pixelBuffer) }

            guard let box = lastBox else {
                // Face lost (beyond the tracker's hysteresis): drop the partial window so we
                // never stitch across a temporal gap. The HR reading persists briefly.
                ring.reset()
                frameCounter = 0
                if doDetect {
                    emitFace(FaceState(faceFound: false, faceBox: nil, bufferSize: bufferSize,
                                       lowLight: false, framesFilled: ring.filled, capacity: AppConfig.framesIn))
                }
                return
            }

            guard let cropped = FrameCropper.crop(pixelBuffer, roi: box) else { return }
            let lowLight = cropped.meanLuma < AppConfig.lowLumaThreshold
            ring.append(cropped.planarRGB)
            frameCounter &+= 1

            if doDetect {
                emitFace(FaceState(faceFound: true, faceBox: box, bufferSize: bufferSize,
                                   lowLight: lowLight, framesFilled: ring.filled, capacity: AppConfig.framesIn))
            }

            if ring.isFull, frameCounter >= AppConfig.stride, !isBusy() {
                frameCounter = 0
                setBusy(true)
                guard let snapshot = ring.snapshot() else { setBusy(false); return }
                inferenceQueue.async { [weak self] in self?.runInference(snapshot) }
            }
        }
    }

    // MARK: - Inference (inference queue)

    private func runInference(_ snapshot: [Float]) {
        defer { setBusy(false) }
        // Pool drains the large input/output tensors and the CoreML/Espresso runtime's
        // autoreleased intermediate buffers at the end of every window, so memory returns
        // to baseline between inferences instead of growing across the session.
        autoreleasepool {
            guard let tensor = TensorBuilder.build(snapshot) else { return }
            let chunkOut: [Float]
            do {
                chunkOut = try model.infer(tensor)   // `chunk` raw rPPG samples
            } catch {
                return   // transient failure: skip this window
            }

            inferenceCount += 1
            if inferenceCount % 10 == 1 {
                MemoryProbe.log("infer #\(inferenceCount)")
            }

            // Stitch this chunk's samples into the rolling analysis buffer.
            rawWaveform.append(contentsOf: chunkOut)
            if rawWaveform.count > AppConfig.analysisSamples {
                rawWaveform.removeFirst(rawWaveform.count - AppConfig.analysisSamples)
            }

            let progress = min(Double(rawWaveform.count) / Double(AppConfig.minAnalysisSamples), 1)

            var shownBPM = bpmFilter.value
            var quality = 0.0
            // Estimate over the stitched buffer once there's enough signal (~2 s).
            if rawWaveform.count >= 60, let result = HeartRateEstimator.estimate(rawWaveform, fs: AppConfig.fps) {
                displayWaveform = Array(result.filtered.suffix(AppConfig.analysisSamples))
                quality = result.quality
                // Update the shown HR on ANY physiological peak once warmed up — the
                // median+EMA filter handles noise, and `quality` drives the badge separately.
                // (Don't hard-gate the number on quality, or a weak-but-valid pulse shows nothing.)
                if rawWaveform.count >= AppConfig.minAnalysisSamples,
                   result.bpm >= AppConfig.minBPM, result.bpm <= AppConfig.maxBPM {
                    shownBPM = bpmFilter.update(result.bpm)
                }
            }

            let update = VitalsUpdate(bpm: shownBPM, quality: quality,
                                      waveform: displayWaveform, latencyMs: model.lastLatencyMs,
                                      warmupProgress: progress)
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.delegate?.pipeline(self, didUpdateVitals: update)
            }
        }
    }

    // MARK: - Helpers

    private func emitFace(_ state: FaceState) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.delegate?.pipeline(self, didUpdateFace: state)
        }
    }

    private func isBusy() -> Bool { busyLock.lock(); defer { busyLock.unlock() }; return busy }
    private func setBusy(_ v: Bool) { busyLock.lock(); busy = v; busyLock.unlock() }
}
