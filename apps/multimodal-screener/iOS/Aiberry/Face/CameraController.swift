import AVFoundation
import CoreVideo

/// Video-only front-camera capture. Delivers BGRA pixel buffers, already rotated
/// upright (portrait) and **not** mirrored, on a serial queue — throttled to
/// `AppConfig.Face.inferenceHz` so the ANE/thermals stay sane across a multi-minute
/// session. Audio is owned separately by `SessionAudio` + `AudioRecorder`; this only
/// touches the capture session, never AVAudioSession.
final class CameraController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    /// Live preview layer (mirrored, natural selfie) for `CameraPreviewView`.
    let previewLayer = AVCaptureVideoPreviewLayer()

    /// Called on the capture queue for each frame that passes the throttle. The
    /// consumer (FaceEmotionModel) applies its own single-in-flight gate.
    var onFrame: ((CVPixelBuffer) -> Void)?

    @Published var authorized = false
    @Published var running = false

    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "aiberry.camera.video")
    private var configured = false
    private var lastFrameTime: CFTimeInterval = 0

    override init() {
        super.init()
        previewLayer.session = session
        previewLayer.videoGravity = .resizeAspectFill
    }

    /// Request camera permission, configure once, and start the session.
    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            authorized = true
            beginSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] ok in
                DispatchQueue.main.async {
                    self?.authorized = ok
                    if ok { self?.beginSession() }
                }
            }
        default:
            authorized = false
        }
    }

    func stop() {
        queue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
            DispatchQueue.main.async { self.running = false }
        }
    }

    private func beginSession() {
        queue.async { [weak self] in
            guard let self else { return }
            if !self.configured { self.configure() }
            guard self.configured, !self.session.isRunning else { return }
            self.session.startRunning()
            DispatchQueue.main.async { self.running = true }
        }
    }

    private func configure() {
        session.beginConfiguration()
        session.sessionPreset = .vga640x480   // plenty for a 224² face crop; light on ANE/thermals

        // CRITICAL: don't let the capture session reconfigure/seize the app audio
        // session — `SessionAudio` owns it so the mic (AVAudioEngine) keeps running
        // alongside the camera. Without this the capture session stomps our category.
        session.automaticallyConfiguresApplicationAudioSession = false

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else { session.commitConfiguration(); return }
        session.addOutput(output)

        // Analysis path: upright portrait, NOT mirrored → Vision orientation .up,
        // crop coordinates are trustworthy. NOTE: `isVideoMirrored` throws unless
        // `automaticallyAdjustsVideoMirroring` is disabled first — that exception is
        // a classic crash-on-entry.
        if let conn = output.connection(with: .video) {
            if conn.isVideoOrientationSupported { conn.videoOrientation = .portrait }
            if conn.isVideoMirroringSupported {
                conn.automaticallyAdjustsVideoMirroring = false
                conn.isVideoMirrored = false
            }
        }
        // Preview path: mirrored selfie so the self-view looks natural.
        if let pConn = previewLayer.connection {
            if pConn.isVideoOrientationSupported { pConn.videoOrientation = .portrait }
            if pConn.isVideoMirroringSupported {
                pConn.automaticallyAdjustsVideoMirroring = false
                pConn.isVideoMirrored = true
            }
        }

        session.commitConfiguration()
        configured = true
    }

    // MARK: - Frame delivery (throttled)

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        let now = CACurrentMediaTime()
        let interval = thermalInterval()
        guard now - lastFrameTime >= interval else { return }
        lastFrameTime = now
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        onFrame?(pixelBuffer)
    }

    /// Back off the FER cadence when the device is heating up.
    private func thermalInterval() -> CFTimeInterval {
        switch ProcessInfo.processInfo.thermalState {
        case .serious, .critical: return AppConfig.Face.frameInterval * 2
        default:                  return AppConfig.Face.frameInterval
        }
    }
}
