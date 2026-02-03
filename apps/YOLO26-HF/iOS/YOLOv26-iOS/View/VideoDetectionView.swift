import SwiftUI
import AVKit
import Vision

struct VideoDetectionView: View {
    let videoURL: URL
    @ObservedObject var detector: YOLOv26Model
    
    @State private var player: AVPlayer?
    @State private var videoOutput: AVPlayerItemVideoOutput?
    @State private var curBoxes: [BoundingBox] = []
    @State private var processingTime: Double = 0
    @State private var isPlaying = false
    
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
                // We need to map boxes (0-1) to the actual rendered video frame rect within the view.
                // VideoPlayer usually fits content.
                // We will assume "Fit" logic.
                // Correct mapping requires knowing the video aspect ratio vs view aspect ratio.
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
        
        // Setup Output
        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
        ]
        videoOutput = AVPlayerItemVideoOutput(pixelBufferAttributes: settings)
        item.add(videoOutput!)
        
        player = AVPlayer(playerItem: item)
        detector.isModelLoaded = true // Ensure model is ready (handled in parent)
    }
    
    private func processFrame() {
        guard let player = player,
              let item = player.currentItem,
              let output = videoOutput,
              isPlaying else { return }
        
        let currentTime = item.currentTime()
        
        if output.hasNewPixelBuffer(forItemTime: currentTime) {
            if let pixelBuffer = output.copyPixelBuffer(forItemTime: currentTime, itemTimeForDisplay: nil) {
                // Process
                runDetection(pixelBuffer: pixelBuffer)
            }
        }
    }
    
    private func runDetection(pixelBuffer: CVPixelBuffer) {
        // Convert CVPixelBuffer -> UIImage
        // Note: This is inefficient but reuses existing 'detect(image:)'
        // Optimization: Implementing direct CVPixelBuffer -> Tensor in YOLOv26Model would be better.
        // For now, fast prototype:
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        let image = UIImage(cgImage: cgImage)
        
        detector.detect(image: image) { results, time in
            self.curBoxes = results
            self.processingTime = time
        }
    }
    
    // Basic helper to get video size for Aspect Ratio calc
    private func getVideoSize() -> CGSize {
        guard let track = player?.currentItem?.asset.tracks(withMediaType: .video).first else { return CGSize(width: 1, height: 1) }
        return track.naturalSize
    }
}

struct VideoBoundingBoxOverlay: View {
    let boxes: [BoundingBox]
    let geometry: GeometryProxy
    let videoSize: CGSize
    
    var body: some View {
        // Calculate the actual frame of the video rendered inside the view (Aspect Fit)
        let viewW = geometry.size.width
        let viewH = geometry.size.height
        
        let videoRatio = videoSize.width / videoSize.height
        let viewRatio = viewW / viewH
        
        var renderW: CGFloat = viewW
        var renderH: CGFloat = viewH
        var offsetX: CGFloat = 0
        var offsetY: CGFloat = 0
        
        if viewRatio > videoRatio {
            // View is wider than video -> fit height, pillarbox
            renderH = viewH
            renderW = renderH * videoRatio
            offsetX = (viewW - renderW) / 2
        } else {
            // View is taller than video -> fit width, letterbox
            renderW = viewW
            renderH = renderW / videoRatio
            offsetY = (viewH - renderH) / 2
        }
        
        return ZStack(alignment: .topLeading) {
            ForEach(boxes) { box in
                let rect = box.rect
                // Scale normalized coords to Rendered Rect
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
                    .offset(x: x, y: y - 14) // Position label above box
            }
        }
    }
}
