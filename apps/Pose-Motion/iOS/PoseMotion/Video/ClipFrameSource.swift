import AVFoundation
import CoreVideo
import QuartzCore

/// Decodes the bundled clip with AVAssetReader and feeds BGRA frames to the
/// inference queue. Two pacing modes:
/// - `.benchmark`: the next frame is decoded only after the previous one finishes
///   processing, so measured FPS is the true sustainable pipeline throughput.
/// - `.realtime`: frames are paced to their presentation timestamps (late frames
///   are dropped), matching how the clip would play live.
/// Loops at end of file.
final class ClipFrameSource {
    enum Mode: String, CaseIterable, Identifiable {
        case benchmark = "Benchmark"
        case realtime = "Realtime"
        var id: String { rawValue }
    }

    /// Called on the pump queue for every frame to process.
    var onFrame: ((CVPixelBuffer, CMTime) -> Void)?

    private let url: URL
    private let queue: DispatchQueue
    private let mode: Mode
    private var reader: AVAssetReader?
    private var output: AVAssetReaderTrackOutput?
    private var running = false
    private var wallStart: CFTimeInterval = 0

    init(url: URL, mode: Mode, queue: DispatchQueue) {
        self.url = url
        self.mode = mode
        self.queue = queue
    }

    func start() {
        queue.async { [weak self] in
            guard let self, !self.running else { return }
            self.running = true
            self.openReader()
            self.step()
        }
    }

    func stop() {
        queue.async { [weak self] in
            guard let self else { return }
            self.running = false
            self.reader?.cancelReading()
            self.reader = nil
            self.output = nil
        }
    }

    // MARK: - Pump (all on `queue`)

    private func openReader() {
        reader?.cancelReading()
        reader = nil
        output = nil

        let asset = AVAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first,
              let reader = try? AVAssetReader(asset: asset) else {
            running = false
            return
        }
        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            running = false
            return
        }
        reader.add(output)
        reader.startReading()
        self.reader = reader
        self.output = output
        wallStart = CACurrentMediaTime()
    }

    private func step() {
        queue.async { [weak self] in
            guard let self, self.running else { return }

            guard let sample = self.output?.copyNextSampleBuffer() else {
                self.openReader()   // end of clip (or reader error): loop
                if self.running { self.step() }
                return
            }
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else {
                self.step()
                return
            }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)

            if self.mode == .realtime {
                let due = self.wallStart + pts.seconds
                let now = CACurrentMediaTime()
                if now < due {
                    Thread.sleep(forTimeInterval: due - now)
                } else if now - due > 0.05 {
                    self.step()   // more than a frame late: drop to catch up
                    return
                }
            }

            self.onFrame?(pixelBuffer, pts)
            self.step()
        }
    }
}
