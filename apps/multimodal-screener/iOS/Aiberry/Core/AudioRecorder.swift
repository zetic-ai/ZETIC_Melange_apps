import AVFoundation
import Combine

/// Microphone recorder producing 16 kHz mono Float32 audio.
///
/// Two modes:
///   • auto-stop  — `record(autoStopSeconds: 3)` finishes itself after N seconds.
///   • manual     — `record(autoStopSeconds: nil)` records until `stop()` is called
///                  (with a 30 s safety cap).
///
/// `ownsSession` controls AVAudioSession lifecycle: standalone callers leave it
/// `true` (the recorder configures + activates the session). Inside the guided
/// check-in a single `SessionAudio` owner configures the session once and keeps it
/// active across questions (so a live `AVCaptureSession` isn't disrupted by
/// per-clip activate/deactivate), so the check-in sets `ownsSession = false`.
final class AudioRecorder: ObservableObject {
    @Published var isRecording = false
    @Published var level: Float = 0          // 0…1, drives the meter UI

    /// When false, the recorder will not touch AVAudioSession (an external owner does).
    var ownsSession = true

    private let engine = AVAudioEngine()
    private let targetSampleRate = Double(AppConfig.sampleRate)
    private let maxSamples = AppConfig.sampleRate * 30   // hard safety cap

    private var captured: [Float] = []
    private var completion: (([Float]) -> Void)?
    private var autoStopSamples: Int?
    private var tapInstalled = false
    private var isActive = false            // synchronous lifecycle guard (main thread only)

    /// Start recording. Pass `autoStopSeconds: nil` for tap-to-stop.
    /// Safe to call repeatedly; ignored while a recording is already in flight.
    func record(autoStopSeconds: Double?, completion: @escaping ([Float]) -> Void) {
        guard !isActive else { return }
        isActive = true
        self.completion = completion
        self.autoStopSamples = autoStopSeconds.map { Int(Double(AppConfig.sampleRate) * $0) }
        requestPermission { [weak self] granted in
            guard let self else { return }
            if granted {
                self.start()
            } else {
                self.isActive = false
                self.completion = nil
            }
        }
    }

    /// Finish a manual recording and deliver what was captured.
    func stop() {
        guard isActive else { return }
        finish(success: true)
    }

    private func requestPermission(_ cb: @escaping (Bool) -> Void) {
        if #available(iOS 17.0, *) {
            AVAudioApplication.requestRecordPermission { ok in DispatchQueue.main.async { cb(ok) } }
        } else {
            AVAudioSession.sharedInstance().requestRecordPermission { ok in DispatchQueue.main.async { cb(ok) } }
        }
    }

    private func start() {
        captured.removeAll(keepingCapacity: true)

        // Clean slate so repeated recordings work reliably.
        if engine.isRunning { engine.stop() }
        if tapInstalled { engine.inputNode.removeTap(onBus: 0); tapInstalled = false }
        engine.reset()

        // Standalone: configure + activate the session here. Inside the check-in the
        // shared SessionAudio owner has already done this and keeps it active.
        if ownsSession {
            let session = AVAudioSession.sharedInstance()
            try? session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker])
            try? session.setActive(true, options: [])
            try? session.overrideOutputAudioPort(.speaker)
        }

        let input = engine.inputNode
        let hwFormat = input.outputFormat(forBus: 0)
        guard hwFormat.sampleRate > 0,
              let dstFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                            sampleRate: targetSampleRate,
                                            channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: hwFormat, to: dstFormat)
        else {
            finish(success: false)
            return
        }

        input.installTap(onBus: 0, bufferSize: 2048, format: hwFormat) { [weak self] buffer, _ in
            self?.process(buffer, converter: converter, dstFormat: dstFormat)
        }
        tapInstalled = true

        engine.prepare()
        do {
            try engine.start()
            DispatchQueue.main.async { self.isRecording = true }
        } catch {
            print("AudioRecorder: engine failed to start — \(error)")
            finish(success: false)
        }
    }

    private func process(_ buffer: AVAudioPCMBuffer,
                         converter: AVAudioConverter,
                         dstFormat: AVAudioFormat) {
        let ratio = dstFormat.sampleRate / converter.inputFormat.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio + 32)
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: dstFormat, frameCapacity: capacity) else { return }

        var error: NSError?
        var supplied = false
        converter.convert(to: outBuffer, error: &error) { _, status in
            if supplied { status.pointee = .noDataNow; return nil }
            supplied = true
            status.pointee = .haveData
            return buffer
        }
        guard error == nil, let channel = outBuffer.floatChannelData?[0] else { return }

        let n = Int(outBuffer.frameLength)
        guard n > 0 else { return }
        let samples = UnsafeBufferPointer(start: channel, count: n)
        captured.append(contentsOf: samples)

        var sumSq: Float = 0
        for s in samples { sumSq += s * s }
        let rms = (sumSq / Float(n)).squareRoot()
        DispatchQueue.main.async { self.level = min(1, rms * 8) }

        if let limit = autoStopSamples, captured.count >= limit {
            DispatchQueue.main.async { self.finish(success: true) }
        } else if captured.count >= maxSamples {
            DispatchQueue.main.async { self.finish(success: true) }
        }
    }

    private func finish(success: Bool) {
        guard isActive else { return }       // idempotent: only the first call runs
        isActive = false
        if tapInstalled { engine.inputNode.removeTap(onBus: 0); tapInstalled = false }
        if engine.isRunning { engine.stop() }
        if ownsSession {
            try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        }

        let deliver = autoStopSamples.map { Array(captured.prefix($0)) } ?? captured
        let cb = completion
        completion = nil
        DispatchQueue.main.async {
            self.isRecording = false
            self.level = 0
            if success, !deliver.isEmpty { cb?(deliver) }
        }
    }
}
