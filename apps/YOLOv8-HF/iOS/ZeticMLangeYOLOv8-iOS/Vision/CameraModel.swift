import SwiftUI
import AVFoundation
import Vision

import ZeticMLange

class CameraModel: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    @Published var session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let processingQueue = DispatchQueue(label: "image_processing_queue", qos: .userInitiated)
    private let fpsQueue = DispatchQueue(label: "fps_queue", qos: .userInitiated)
    private var contextPool: CGContext?
    private let contextQueue = DispatchQueue(label: "context_queue")
    public var resolution = CGSize(width: 640, height: 480)
    private let position: AVCaptureDevice.Position
    private let mode: CameraMode
    
    private let context: CIContext
    
    private var isProcessing: Bool = false
    
    
    @Published var frame: CameraFrame? = nil
    @Published var timestamp: CMTime? = nil
    @Published var image: UIImage? = nil
    
    init(_ position: AVCaptureDevice.Position, _ mode: CameraMode) {
        let contextOptions = [CIContextOption.useSoftwareRenderer: false]
        self.context = CIContext(options: contextOptions)
        self.mode = mode
        self.position = position
        
        super.init()
    }
    
    func close() {
        session.stopRunning()
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        if (mode == .frame) {
            captureFrame(sampleBuffer)
        }
        
        if(mode == .image) {
            captureImage(sampleBuffer)
        }
    }
    
    private func captureFrame(_ sampleBuffer: CMSampleBuffer) {
        autoreleasepool {
            let currentTimestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            
            DispatchQueue.main.async {
                self.frame = CameraFrame(from: sampleBuffer)
                self.timestamp = currentTimestamp
            }
        }
    }
    
    private func captureImage(_ sampleBuffer: CMSampleBuffer) {
        autoreleasepool {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                return
            }
            
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            
            let transform = CGAffineTransform(scaleX: -1, y: 1)
            
            let flippedImage = ciImage.transformed(by: transform)
            
            
            guard let cgImage = context.createCGImage(flippedImage, from: flippedImage.extent) else {
                return
            }
            
            let uiImage = UIImage(cgImage: cgImage)
            
            DispatchQueue.main.async {
                self.image = uiImage
            }
        }
    }
    
    private func setupCamera() {
        do {
            session.beginConfiguration()
            
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) else {
                print("Error: Unable to access the back camera.")
                session.commitConfiguration()
                return
            }
            
            var selectedFormat: AVCaptureDevice.Format?
            var maxFrameRate: Double = 0
            
            for format in device.formats {
                let formatDescription = format.formatDescription
                let dimensions = CMVideoFormatDescriptionGetDimensions(formatDescription)
                
                print(format)
                for range in format.videoSupportedFrameRateRanges {
                    if range.maxFrameRate == 120 {
                        selectedFormat = format
                        maxFrameRate = range.maxFrameRate
                        break
                    }
                }
                
                if selectedFormat != nil {
                    break
                }
            }
            if selectedFormat == nil {
                selectedFormat = device.activeFormat
                maxFrameRate = selectedFormat!.videoSupportedFrameRateRanges.first!.maxFrameRate
            }
            
            if let selectedFormat = selectedFormat {
                try device.lockForConfiguration()
                device.activeFormat = selectedFormat
                device.activeVideoMinFrameDuration = CMTimeMake(value: 1, timescale: Int32(maxFrameRate))
                device.activeVideoMaxFrameDuration = CMTimeMake(value: 1, timescale: Int32(maxFrameRate))
                device.unlockForConfiguration()
                print("Selected format: \(selectedFormat)")
                print("Configured min frame duration: \(device.activeVideoMinFrameDuration)")
                print("Configured max frame duration: \(device.activeVideoMaxFrameDuration)")
                let configuredFPS = Double(device.activeVideoMaxFrameDuration.timescale) / Double(device.activeVideoMaxFrameDuration.value)
                print("Configured FPS: \(configuredFPS)")
            } else {
                print("Error: 240fps not supported on this device at the desired resolution.")
                session.commitConfiguration()
                return
            }
            
            let input = try AVCaptureDeviceInput(device: device)
            
            // Set session preset to input priority
            session.sessionPreset = .inputPriority
            
            if session.canAddInput(input) {
                session.addInput(input)
            } else {
                print("Error: Cannot add input to session.")
                session.commitConfiguration()
                return
            }
            
            // Configure video output
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
            ]
            videoOutput.setSampleBufferDelegate(self, queue: processingQueue)
            
            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
            } else {
                print("Error: Cannot add output to session.")
                session.commitConfiguration()
                return
            }
            
            // Set the video orientation and disable stabilization
            if let connection = videoOutput.connection(with: .video) {
                connection.videoOrientation = .portrait
                connection.isVideoMirrored = false
                if connection.isVideoStabilizationSupported {
                    connection.preferredVideoStabilizationMode = .off
                }
            }
            
            session.commitConfiguration()
            
            // Start running the session
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.session.startRunning()
                self?.resolution = CGSize(
                    width: CGFloat(device.activeFormat.formatDescription.dimensions.width),
                    height: CGFloat(device.activeFormat.formatDescription.dimensions.height)
                )
            }
        } catch {
            print("Camera setup error: \(error)")
            session.commitConfiguration()
        }
    }
    
    func checkCameraPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            setupCamera()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                if granted {
                    DispatchQueue.main.async {
                        self?.setupCamera()
                    }
                }
            }
        default:
            break
        }
    }
}
