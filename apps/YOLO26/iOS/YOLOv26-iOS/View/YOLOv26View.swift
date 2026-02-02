import SwiftUI
import UIKit
import PhotosUI
import AVKit
import Vision
import UniformTypeIdentifiers

// MARK: - ImagePicker
struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    @Environment(\.presentationMode) var presentationMode

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .photoLibrary
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: ImagePicker

        init(parent: ImagePicker) {
            self.parent = parent
        }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage
            }
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}

// MARK: - VideoPicker
struct VideoPicker: UIViewControllerRepresentable {
    @Binding var videoURL: URL?
    @Binding var isProcessing: Bool
    @Environment(\.presentationMode) var presentationMode

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .videos
        config.selectionLimit = 1
        
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: VideoPicker

        init(parent: VideoPicker) {
            self.parent = parent
        }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            // Dismissal is deferred until after handling the file
            // parent.presentationMode.wrappedValue.dismiss()
            
            guard let provider = results.first?.itemProvider,
                  provider.hasItemConformingToTypeIdentifier(UTType.movie.identifier) else {
                parent.presentationMode.wrappedValue.dismiss()
                return
            }
            
            // Start Processing State
            parent.isProcessing = true
            
            // Load file representation
            provider.loadFileRepresentation(forTypeIdentifier: UTType.movie.identifier) { url, error in
                defer {
                    DispatchQueue.main.async {
                        self.parent.isProcessing = false
                        self.parent.presentationMode.wrappedValue.dismiss()
                    }
                }
                
                if let error = error {
                    print("Error loading video: \(error.localizedDescription)")
                    return
                }
                
                guard let url = url else { return }
                
                // Copy to a accessible temp location
                let fileName = "temp_video_\(Date().timeIntervalSince1970).mov"
                let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
                
                do {
                    if FileManager.default.fileExists(atPath: tempURL.path) {
                        try FileManager.default.removeItem(at: tempURL)
                    }
                    try FileManager.default.copyItem(at: url, to: tempURL)
                    
                    DispatchQueue.main.async {
                        self.parent.videoURL = tempURL
                    }
                } catch {
                    print("Error copying video file: \(error.localizedDescription)")
                }
            }
        }
    }
}

// MARK: - VideoDetectionView
struct VideoDetectionView: View {
    let videoURL: URL
    @ObservedObject var detector: YOLOv26Model
    
    @State private var player: AVPlayer?
    @State private var videoOutput: AVPlayerItemVideoOutput?
    @State private var curBoxes: [BoundingBox] = []
    @State private var processingTime: Double = 0
    @State private var isPlaying = false
    @State private var videoOrientation: CGImagePropertyOrientation = .up
    
    // Timer for polling frames
    let timer = Timer.publish(every: 0.03, on: .main, in: .common).autoconnect() // ~30 FPS check
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Video Player
                if let player = player {
                    VideoPlayer(player: player)
                        .onAppear {
                            player.play()
                            isPlaying = true
                        }
                        .onDisappear {
                            player.pause()
                        }
                }
                
                // Overlay Boxes
                VideoBoundingBoxOverlay(boxes: curBoxes, geometry: geometry, videoSize: getVideoSize())
            }
        }
        .onAppear {
            setupPlayer()
        }
        .onReceive(timer) { _ in
            processFrame()
        }
    }
    
    private func setupPlayer() {
        let item = AVPlayerItem(url: videoURL)
        
        // Determine Orientation
        let asset = AVAsset(url: videoURL)
        if let track = asset.tracks(withMediaType: .video).first {
             self.videoOrientation = getOrientation(from: track.preferredTransform)
        }
        
        // Setup Output
        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
        ]
        videoOutput = AVPlayerItemVideoOutput(pixelBufferAttributes: settings)
        item.add(videoOutput!)
        
        player = AVPlayer(playerItem: item)
        detector.isModelLoaded = true
    }
    
    private func getOrientation(from transform: CGAffineTransform) -> CGImagePropertyOrientation {
        if transform.a == 0 && transform.b == 1.0 && transform.c == -1.0 && transform.d == 0 {
            return .right
        } else if transform.a == 0 && transform.b == -1.0 && transform.c == 1.0 && transform.d == 0 {
            return .left
        } else if transform.a == -1.0 && transform.b == 0 && transform.c == 0 && transform.d == -1.0 {
            return .down
        } else {
            return .up
        }
    }
    
    private func processFrame() {
        guard let player = player,
              let item = player.currentItem,
              let output = videoOutput,
              isPlaying else { return }
        
        let currentTime = item.currentTime()
        
        if output.hasNewPixelBuffer(forItemTime: currentTime) {
            if let pixelBuffer = output.copyPixelBuffer(forItemTime: currentTime, itemTimeForDisplay: nil) {
                runDetection(pixelBuffer: pixelBuffer)
            }
        }
    }
    
    private func runDetection(pixelBuffer: CVPixelBuffer) {
        var ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        // Apply Orientation to simulate 'Upright' view before inference
        ciImage = ciImage.oriented(self.videoOrientation)
        
        // Note: CIImage.extent changes after orientation
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        let image = UIImage(cgImage: cgImage)
        
        detector.detect(image: image) { results, time in
            self.curBoxes = results
            self.processingTime = time
        }
    }
    
    private func getVideoSize() -> CGSize {
        guard let track = player?.currentItem?.asset.tracks(withMediaType: .video).first else { return CGSize(width: 1, height: 1) }
        let size = track.naturalSize
        let transform = track.preferredTransform
        
        if size.width == transform.tx && size.height == transform.ty {
            return size
        } else if transform.tx == 0 && transform.ty == 0 {
            return size
        } else if transform.tx == 0 && transform.ty == size.width {
            return CGSize(width: size.height, height: size.width)
        } else if transform.tx == size.height && transform.ty == 0 {
            return CGSize(width: size.height, height: size.width)
        } else {
            return size
        }
    }
}

struct VideoBoundingBoxOverlay: View {
    let boxes: [BoundingBox]
    let geometry: GeometryProxy
    let videoSize: CGSize
    
    var body: some View {
        let viewW = geometry.size.width
        let viewH = geometry.size.height
        
        let videoRatio = videoSize.width / videoSize.height
        let viewRatio = viewW / viewH
        
        var renderW: CGFloat = viewW
        var renderH: CGFloat = viewH
        var offsetX: CGFloat = 0
        var offsetY: CGFloat = 0
        
        if viewRatio > videoRatio {
            renderH = viewH
            renderW = renderH * videoRatio
            offsetX = (viewW - renderW) / 2
        } else {
            renderW = viewW
            renderH = renderW / videoRatio
            offsetY = (viewH - renderH) / 2
        }
        
        return ZStack(alignment: .topLeading) {
            ForEach(boxes) { box in
                let rect = box.rect
                let x = rect.minX * renderW + offsetX
                let y = rect.minY * renderH + offsetY
                let w = rect.width * renderW
                let h = rect.height * renderH
                
                let color = Color.color(for: box.classIndex)
                
                Rectangle()
                    .path(in: CGRect(x: x, y: y, width: w, height: h))
                    .stroke(color, lineWidth: 2)
                
                Text("\(box.label) \(String(format: "%.2f", box.score))")
                    .font(.system(size: 10, weight: .bold))
                    .padding(2)
                    .background(color)
                    .foregroundColor(.black)
                    .offset(x: x, y: y - 14)
            }
        }
    }
}

// MARK: - ContentView
struct ContentView: View {
    enum DetectionMode {
        case image
        case video
        case camera
    }

    @State private var image: UIImage?
    @State private var videoURL: URL? // For Video Mode
    @State private var isProcessingVideo = false // Loading State
    @State private var showingImagePicker = false
    @State private var showingVideoPicker = false
    
    // @State private var boxes: [BoundingBox] = [] // Removed, using detector.boxes
    @State private var processingTime: Double = 0
    @StateObject private var detector = YOLOv26Model()
    
    @State private var detectionMode: DetectionMode = .image
    
    var body: some View {
        NavigationView {
            ZStack {
                if !detector.isModelLoaded {
                    VStack {
                        ProgressView()
                            .scaleEffect(2.0)
                        Text("Loading Model...")
                            .padding(.top, 20)
                    }
                } else {
                    VStack {
                        // Label
                        Text("(Powered by MLange)")
                            .font(.caption)
                            .foregroundColor(.gray)
                            .padding(.top, 4)

                        // Mode Selector
                        Picker("Mode", selection: $detectionMode) {
                            Text("Photo").tag(DetectionMode.image)
                            Text("Video").tag(DetectionMode.video)
                            Text("Camera").tag(DetectionMode.camera)
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        .padding()
                        
                        // Content Area
                        if detectionMode == .camera {
                            ZStack {
                                CameraView(detector: detector)
                                    .edgesIgnoringSafeArea(.all)
                                
                                // Overlay for Camera
                                GeometryReader { geometry in
                                    self.boundingBoxesView(geometry: geometry)
                                }
                            }
                        } else if detectionMode == .video {
                            if isProcessingVideo {
                                VStack {
                                    ProgressView()
                                        .scaleEffect(1.5)
                                    Text("Processing Video...")
                                        .padding(.top, 10)
                                }
                                .frame(maxHeight: 500)
                            } else if let videoURL = videoURL {
                                VideoDetectionView(videoURL: videoURL, detector: detector)
                                    .frame(maxHeight: 500)
                            } else {
                                Text("Select a video to start detection")
                                    .foregroundColor(.gray)
                                    .padding()
                            }
                        } else {
                            // Image Mode
                            if let image = image {
                                ZStack {
                                    Image(uiImage: image)
                                        .resizable()
                                        .aspectRatio(contentMode: .fit)
                                        .overlay(
                                            GeometryReader { geometry in
                                                self.boundingBoxesView(geometry: geometry)
                                            }
                                        )
                                }
                                .frame(maxHeight: 500)
                                .padding()
                            } else {
                                Text("Select an image to start detection")
                                    .foregroundColor(.gray)
                                    .padding()
                            }
                        }
                        
                        if processingTime > 0 {
                            Text(String(format: "Processing Time: %.3f s", processingTime))
                                .font(.caption)
                                .padding(.top, 5)
                            
                            Text(detector.debugText)
                                .font(.caption2)
                                .foregroundColor(.red)
                                .padding(.top, 2)
                        }
                        
                        Spacer()
                        
                        // Controls
                        if detectionMode == .image {
                            Button(action: {
                                showingImagePicker = true
                            }) {
                                Text("Select Image")
                                    .font(.headline)
                                    .padding()
                                    .frame(minWidth: 200)
                                    .background(Color.blue)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            }
                            .padding(.bottom, 10)
                            
                            if !detector.boxes.isEmpty {
                                Button(action: {
                                    saveImage()
                                }) {
                                    Text("Save Result")
                                        .font(.headline)
                                        .padding()
                                        .frame(minWidth: 200)
                                        .background(Color.green)
                                        .foregroundColor(.white)
                                        .cornerRadius(10)
                                }
                                .padding(.bottom, 20)
                            }
                        } else if detectionMode == .video {
                            Button(action: {
                                showingVideoPicker = true
                            }) {
                                Text("Select Video")
                                    .font(.headline)
                                    .padding()
                                    .frame(minWidth: 200)
                                    .background(isProcessingVideo ? Color.gray : Color.purple)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            }
                            .padding(.bottom, 20)
                            .disabled(isProcessingVideo)
                        }
                    }
                    .navigationTitle("YOLOv26 Detection")
                    .sheet(isPresented: $showingImagePicker, onDismiss: detectObjects) {
                        ImagePicker(image: $image)
                    }
                    .sheet(isPresented: $showingVideoPicker) {
                        VideoPicker(videoURL: $videoURL, isProcessing: $isProcessingVideo)
                    }
                }
            }
        }
    }

    
    func detectObjects() {
        guard let image = image else { return }
        
        // Clear previous results
        detector.boxes = []
        self.processingTime = 0
        
        detector.detect(image: image) { results, time in
            // self.boxes = results // Updated via Published property
            self.processingTime = time
        }
    }
    
    func boundingBoxesView(geometry: GeometryProxy) -> some View {
        // Camera Config: 640x480 preset in Portrait -> 480x640
        let videoSize = CGSize(width: 480, height: 640)
        
        let viewW = geometry.size.width
        let viewH = geometry.size.height
        
        let videoRatio = videoSize.width / videoSize.height
        let viewRatio = viewW / viewH
        
        var renderW: CGFloat = viewW
        var renderH: CGFloat = viewH
        var offsetX: CGFloat = 0
        var offsetY: CGFloat = 0
        
        // Aspect Fill Logic (Center Crop)
        // Matches AVCaptureVideoPreviewLayer.videoGravity = .resizeAspectFill
        if viewRatio > videoRatio {
             // View is wider than Video -> Fit Width to View, Crop Top/Bottom? 
             // No, Aspect Fill means coverage.
             // If View is Wider (e.g. iPad maybe?), we must Scale by Width. Video Height will be clipped.
             // Scale = viewW / videoW
             renderW = viewW
             renderH = viewW / videoRatio
             offsetY = (viewH - renderH) / 2
        } else {
             // View is Narrower (Taller, e.g. iPhone) -> Scale by Height. Video Width will be clipped.
             renderH = viewH
             renderW = viewH * videoRatio
             offsetX = (viewW - renderW) / 2
        }
        
        return ForEach(detector.boxes) { box in
            let rect = box.rect
            
            // Map normalized coordinates to the Rendered Video Size
            let x = rect.minX * renderW + offsetX
            let y = rect.minY * renderH + offsetY
            let width = rect.width * renderW
            let height = rect.height * renderH
            
            let color = Color.color(for: box.classIndex)
            
            ZStack(alignment: .topLeading) {
                Rectangle()
                .path(in: CGRect(x: x, y: y, width: width, height: height))
                .stroke(color, lineWidth: 2)
            
                Text("\(box.label) \(String(format: "%.2f", box.score))")
                .font(.system(size: 10, weight: .bold))
                .padding(2)
                .background(color)
                .foregroundColor(.black)
                .offset(x: x, y: y - 14) 
            }
        }
    }
    
    // Draw boxes on high-res image
    func drawDetectionsOnImage(_ original: UIImage, boxes: [BoundingBox]) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: original.size)
        return renderer.image { context in
            original.draw(at: .zero)
            
            let attrs: [NSAttributedString.Key: Any] = [
                .font: UIFont.boldSystemFont(ofSize: max(20, original.size.width / 50)),
                .foregroundColor: UIColor.white,
                .backgroundColor: UIColor.red
            ]
            
            for box in boxes {
                let color = Color.uiColor(for: box.classIndex)
                
                // Update attributes for this box
                var boxAttrs = attrs
                boxAttrs[.backgroundColor] = color
                boxAttrs[.foregroundColor] = UIColor.black
                
                context.cgContext.setLineWidth(max(2, original.size.width / 200))
                context.cgContext.setStrokeColor(color.cgColor)
                
                let rect = box.rect
                let x = rect.minX * original.size.width
                let y = rect.minY * original.size.height
                let w = rect.width * original.size.width
                let h = rect.height * original.size.height
                
                let drawRect = CGRect(x: x, y: y, width: w, height: h)
                context.cgContext.addRect(drawRect)
                context.cgContext.drawPath(using: .stroke)
                
                let label = "\(box.label) \(String(format: "%.2f", box.score))"
                let textPoint = CGPoint(x: x, y: y - max(20, original.size.width / 50))
                (label as NSString).draw(at: textPoint, withAttributes: boxAttrs)
            }
        }
    }
    
    func saveImage() {
        guard let original = image else { return }
        let result = drawDetectionsOnImage(original, boxes: detector.boxes)
        
        let av = UIActivityViewController(activityItems: [result], applicationActivities: nil)
        
        // KeyWindow lookup for SwiftUI
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
            let rootVC = windowScene.windows.first?.rootViewController {
            rootVC.present(av, animated: true, completion: nil)
        }
    }
}

// MARK: - Color Extension
extension Color {
    static func color(for classIndex: Int) -> Color {
        let colors: [Color] = [
            .red, .green, .blue, .orange, .purple, .pink, .yellow, .cyan, .mint, .indigo, .brown, .teal
        ]
        return colors[classIndex % colors.count]
    }
    
    static func uiColor(for classIndex: Int) -> UIColor {
        let colors: [UIColor] = [
            .red, .green, .blue, .orange, .purple, .systemPink, .yellow, .cyan, .systemMint, .systemIndigo, .brown, .systemTeal
        ]
        return colors[classIndex % colors.count]
    }
}
