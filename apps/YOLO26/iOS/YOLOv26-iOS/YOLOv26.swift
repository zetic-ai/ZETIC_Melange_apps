import Foundation
import UIKit
import ZeticMLange
import CoreVideo
import Accelerate
import Accelerate.vImage

// MARK: - CocoClasses
enum CocoClasses {
    // Sourced from apps/YOLOv26/coco.yaml
    static let names: [String] = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    ]
}

// MARK: - BoundingBox
struct BoundingBox: Identifiable {
    let id = UUID()
    let rect: CGRect
    let classIndex: Int
    let score: Float
    let label: String
}

// MARK: - ImageUtils
class ImageUtils {
    static func resize(image: UIImage, to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resizedImage
    }

    static func prepareInput(image: UIImage, targetSize: CGSize = CGSize(width: 640, height: 640)) -> [Float]? {
        // 1. Resize and normalize orientation using UIKit
        // This 'bakes' the orientation into the pixel data ensuring upright image
        guard let resized = resize(image: image, to: targetSize),
              let cgImage = resized.cgImage else { return nil }
        
        let width = Int(targetSize.width)
        let height = Int(targetSize.height)
        let totalPixels = width * height
        
        // Force RGBA (ByteOrder32Big + PremultipliedLast)
        // R G B A
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
        
        guard let context = CGContext(data: nil,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: width * 4,
                                      space: colorSpace,
                                      bitmapInfo: bitmapInfo.rawValue) else { return nil }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))
        
        guard let data = context.data else { return nil }
        
        let ptr = data.bindMemory(to: UInt8.self, capacity: totalPixels * 4)
        var normalizedArray = [Float](repeating: 0.0, count: 3 * totalPixels)
        
        // Extract RGB Planar [3, H, W]
        // Source is Packed RGBA [H, W, 4]
        for i in 0..<totalPixels {
            let offset = i * 4
            let r = Float(ptr[offset]) / 255.0
            let g = Float(ptr[offset + 1]) / 255.0
            let b = Float(ptr[offset + 2]) / 255.0
            
            normalizedArray[i] = r
            normalizedArray[totalPixels + i] = g
            normalizedArray[2 * totalPixels + i] = b
        }
        
        return normalizedArray
    }
}

// MARK: - YOLOv26Model
class YOLOv26Model: ObservableObject {
    private var model: ZeticMLangeModel?
    @Published var isModelLoaded: Bool = false
    
    // Config
    private let targetSize = CGSize(width: 640, height: 640)
    private let confidenceThreshold: Float = 0.25
    private let iouThreshold: Float = 0.45
    private let classCount = 80
    private let totalOutput = 8400 
    
    @Published var debugText: String = ""
    @Published var boxes: [BoundingBox] = []

    // Optimization Buffers (Zero-Allocation)
    private var pixelBuffer: [UInt8]
    private var inputBuffer: [Float]
    private var alphaBuffer: [Float] // Scratch for vImage
    private var resizeContext: CGContext?

    init() {
        self.isModelLoaded = false
        
        // Initialize Buffers (640x640)
        let totalPixels = 640 * 640
        self.pixelBuffer = [UInt8](repeating: 0, count: totalPixels * 4) // RGBA
        self.inputBuffer = [Float](repeating: 0.0, count: totalPixels * 3) // RGB Planar
        self.alphaBuffer = [Float](repeating: 0.0, count: totalPixels)     // Alpha discard
        
        // Create Reusable Context
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
        
        self.resizeContext = self.pixelBuffer.withUnsafeMutableBufferPointer { ptr -> CGContext? in
             return CGContext(
                data: ptr.baseAddress,
                width: 640,
                height: 640,
                bitsPerComponent: 8,
                bytesPerRow: 640 * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo.rawValue
            )
        }

        // Async loading to prevent freezing UI
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                self.model = try ZeticMLangeModel(tokenKey: "YOUR_MLANGE_KEY", name: "Team_ZETIC/YOLOv26", version: 1)
                DispatchQueue.main.async {
                    self.isModelLoaded = true
                }
            } catch {
                print("Failed to load ZeticMLangeModel: \(error)")
            }
        }
    }
    
    func detect(pixelBuffer: CVPixelBuffer, completion: @escaping ([BoundingBox], Double) -> Void) {
        guard let model = model else {
            completion([], 0)
            return
        }
        
        DispatchQueue.global(qos: .userInitiated).async {
            let startTime = CFAbsoluteTimeGetCurrent()
            
            // 1. Resize directly using vImage (skip CoreImage/UIKit)
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
            
            guard let srcBase = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }
            let srcWidth = CVPixelBufferGetWidth(pixelBuffer)
            let srcHeight = CVPixelBufferGetHeight(pixelBuffer)
            let srcBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            
            var srcBuffer = vImage_Buffer(
                data: srcBase,
                height: vImagePixelCount(srcHeight),
                width: vImagePixelCount(srcWidth),
                rowBytes: srcBytesPerRow
            )
            
            // Target Buffer (Reusable self.pixelBuffer)
            let dstWidth = 640
            let dstHeight = 640
            
            self.pixelBuffer.withUnsafeMutableBufferPointer { dstPtr in
                guard let dstBase = dstPtr.baseAddress else { return }
                
                var dstBuffer = vImage_Buffer(
                    data: dstBase,
                    height: vImagePixelCount(dstHeight),
                    width: vImagePixelCount(dstWidth),
                    rowBytes: dstWidth * 4 // RGBA
                )
                
                // Scale (Fast, High Quality)
                // Note: Input is usually BGRA (Camera), Output is RGBA (Model expectation?) or BGRA?
                // Model usually expects RGB.
                // vImage can scale. We handle channel swap in the next step (Planarization).
                vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, vImage_Flags(kvImageNoFlags))
            }
            
            // 2. Normalize & Planarize (Zero Allocation into inputBuffer)
            // Existing logic modified to handle BGRA -> RGB swap if needed
            // CoreVideo Camera default is kCVPixelFormatType_32BGRA
            
            self.pixelBuffer.withUnsafeMutableBufferPointer { pixelPtr in
                self.inputBuffer.withUnsafeMutableBufferPointer { inputPtr in
                     guard let srcBase = pixelPtr.baseAddress,
                           let dstBase = inputPtr.baseAddress else { return }
                     
                     let count = 640 * 640
                     for i in 0..<count {
                         // Input resized buffer is BGRA (from Camera)
                         // We need RGB Planar
                         
                         let offset = i * 4
                         let b = Float(srcBase[offset + 0]) / 255.0
                         let g = Float(srcBase[offset + 1]) / 255.0
                         let r = Float(srcBase[offset + 2]) / 255.0
                         // Alpha at +3 ignored
                         
                         // Planar destinations
                         dstBase[i] = r
                         dstBase[count + i] = g
                         dstBase[2 * count + i] = b
                     }
                }
            }
            
            // 3. Inference
            do {
                let data = Data(buffer: UnsafeBufferPointer(start: self.inputBuffer, count: self.inputBuffer.count))

                let inputTensor = ZeticMLange.Tensor(data: data, dataType: ZeticMLange.BuiltinDataType.float32, shape: [1, 3, 640, 640])
                let inputs: [ZeticMLange.Tensor] = [inputTensor]
                
                let outputs = try model.run(inputs: inputs)
                
                guard let outputTensor = outputs.first else {
                    DispatchQueue.main.async { 
                        self.boxes = []
                        completion([], 0) 
                    }
                    return
                }
                
                // Inspect Shape
                let shape = outputTensor.shape
                let dim1 = shape.count > 1 ? shape[1] : 0
                let dim2 = shape.count > 2 ? shape[2] : 0
                
                var results: [BoundingBox] = []
                var debugInfo = ""
                
                if dim1 == 300 && dim2 == 6 {
                    results = self.postprocessNMS(output: outputTensor, originalSize: CGSize(width: srcWidth, height: srcHeight)) // Use actual src size
                    debugInfo = "Standard Export"
                } else {
                    // Raw logic... (simplified for brevity match)
                     var detectedClassCount = 80
                     var outputAnchors = 8400
                     if dim2 > dim1 { detectedClassCount = dim1 - 4; outputAnchors = dim2 }
                     else { detectedClassCount = dim2 - 4; outputAnchors = dim1 }
                     results = self.postprocess(output: outputTensor, originalSize: CGSize(width: srcWidth, height: srcHeight), classCount: detectedClassCount, outputAnchors: outputAnchors)
                     debugInfo = "Raw Output"
                }
                
                let duration = CFAbsoluteTimeGetCurrent() - startTime
                
                DispatchQueue.main.async {
                    self.debugText = debugInfo
                    self.boxes = results
                    completion(results, duration)
                }
            } catch {
                print("Inference failed: \(error)")
                DispatchQueue.main.async {
                    self.boxes = []
                    completion([], 0)
                }
            }
        }
    }
    
    func detect(image: UIImage, completion: @escaping ([BoundingBox], Double) -> Void) {
        guard let model = model, let context = self.resizeContext, let cgImage = image.cgImage else {
            print("Model/Context not initialized or Invalid Image")
            completion([], 0)
            return
        }
        
        DispatchQueue.global(qos: .userInitiated).async {
            let startTime = CFAbsoluteTimeGetCurrent()
            
            // 1. Resize & Draw to reusable pixel buffer (Zero Allocation)
            // Fix: Handle Image Orientation
            var inputCGImage = cgImage
            if image.imageOrientation != .up {
                // If not upright, use UIKit to normalize orientation into a new 640x640 CGImage
                UIGraphicsBeginImageContextWithOptions(CGSize(width: 640, height: 640), false, 1.0)
                image.draw(in: CGRect(x: 0, y: 0, width: 640, height: 640))
                if let normalized = UIGraphicsGetImageFromCurrentImageContext()?.cgImage {
                    inputCGImage = normalized
                }
                UIGraphicsEndImageContext()
            }
            
            // context.clear(CGRect(x: 0, y: 0, width: 640, height: 640)) 
            context.draw(inputCGImage, in: CGRect(x: 0, y: 0, width: 640, height: 640))
            
            // 2. Normalize & Planarize (Zero Allocation into inputBuffer)
            let width = 640
            let height = 640
            let totalPixels = width * height
            
            // 2. Normalize & Planarize using Accelerate (vImage)
            // Merges: Split Channels, Convert Int->Float, Normalize 0..1
            
            self.pixelBuffer.withUnsafeMutableBufferPointer { pixelPtr in
                self.inputBuffer.withUnsafeMutableBufferPointer { inputPtr in
                    self.alphaBuffer.withUnsafeMutableBufferPointer { alphaPtr in 
                        guard let srcBase = pixelPtr.baseAddress,
                              let dstBase = inputPtr.baseAddress,
                              let alphaBase = alphaPtr.baseAddress else { return }
                              
                        // Manual loop fallback to ensure build success
                        // vImageConvert_RGBA8888toPlanarF requires specific availability and imports
                        // Using unsafe pointers directly is fast enough here
                        
                        let count = 640 * 640
                        for i in 0..<count {
                            // Source is RGBA (4 bytes)
                            let r = Float(srcBase[i * 4 + 0]) / 255.0
                            let g = Float(srcBase[i * 4 + 1]) / 255.0
                            let b = Float(srcBase[i * 4 + 2]) / 255.0
                            // Alpha ignored
                            
                            // Dest is Planar Float (4 bytes)
                            // dstR, dstG, dstB base addresses already offset
                            // We need to cast them to Float because dstBase is float pointer
                            dstBase[i] = r
                            dstBase[count + i] = g
                            dstBase[2 * count + i] = b
                        }
                    }
                }
            }
            
            do {
                // 3. Wrap in Tensor
                // Data(bytesNoCopy:...) avoids another copy if Tensor supports it, 
                // typically we cast the buffer to Data.
                let data = Data(buffer: UnsafeBufferPointer(start: self.inputBuffer, count: self.inputBuffer.count))
                // ... (Original Implementation matches until here)

                let inputTensor = ZeticMLange.Tensor(data: data, dataType: ZeticMLange.BuiltinDataType.float32, shape: [1, 3, 640, 640])
                let inputs: [ZeticMLange.Tensor] = [inputTensor]
                
                let outputs = try model.run(inputs: inputs)
                
                guard let outputTensor = outputs.first else {
                    DispatchQueue.main.async { 
                        self.boxes = []
                        completion([], 0) 
                    }
                    return
                }
                
                
                // Inspect Shape
                let shape = outputTensor.shape
                let dim1 = shape.count > 1 ? shape[1] : 0
                let dim2 = shape.count > 2 ? shape[2] : 0
                
                var results: [BoundingBox] = []
                var debugInfo = ""
                
                // CASE 1: Standard YOLO Output with NMS (e.g. CoreML Export) -> [1, 300, 6]
                // Layout: [Batch, Boxes, 6] where 6 = [x, y, w, h, score, class]
                if dim1 == 300 && dim2 == 6 {
                    results = self.postprocessNMS(output: outputTensor, originalSize: image.size)
                    debugInfo = "Shape: \(shape) | Pre-NMS Output (Standard Export)"
                }
                // CASE 2: Raw YOLO Output -> [1, 84, 8400] (or similar)
                else {
                    var detectedClassCount = 80
                    var outputAnchors = 8400
                    
                    // Heuristic: The dimension closer to 8400 is anchors.
                    if dim2 > dim1 {
                        detectedClassCount = dim1 - 4
                        outputAnchors = dim2
                    } else {
                        detectedClassCount = dim2 - 4
                        outputAnchors = dim1
                    }
                    
                    results = self.postprocess(output: outputTensor, originalSize: image.size, classCount: detectedClassCount, outputAnchors: outputAnchors)
                    debugInfo = "Shape: \(shape) | Classes: \(detectedClassCount) | Anchors: \(outputAnchors) (Raw Output)"
                }
                
                let duration = CFAbsoluteTimeGetCurrent() - startTime
                // print("DEBUG: \(debugInfo)")
                
                DispatchQueue.main.async {
                    self.debugText = debugInfo
                    self.boxes = results
                    completion(results, duration)
                }
            } catch {
                print("Inference failed: \(error)")
                DispatchQueue.main.async {
                    self.debugText = "Error: \(error.localizedDescription)"
                    self.boxes = []
                    completion([], 0)
                }
            }
        }
    }
    
    private func postprocessNMS(output: ZeticMLange.Tensor, originalSize: CGSize) -> [BoundingBox] {
        let data: [Float] = output.data.withUnsafeBytes {
            let buffer = $0.bindMemory(to: Float.self)
            return Array(buffer)
        }
        
        var boxes: [BoundingBox] = []
        let rows = 300
        let cols = 6
        
        for i in 0..<rows {
            let offset = i * cols
            if offset + 5 >= data.count { break }
            
            let score = data[offset + 4]
            if score > confidenceThreshold {
                let classIndex = Int(data[offset + 5])
                
                let x1 = data[offset + 0]
                let y1 = data[offset + 1]
                let x2 = data[offset + 2]
                let y2 = data[offset + 3]
                
                // CoreML Export NMS Output is [x1, y1, x2, y2] in absolute pixels (0-640)
                // Normalize to 0-1
                let rect = CGRect(
                    x: CGFloat(x1) / 640.0,
                    y: CGFloat(y1) / 640.0,
                    width: CGFloat(x2 - x1) / 640.0,
                    height: CGFloat(y2 - y1) / 640.0
                )
                
                let label = (classIndex >= 0 && classIndex < CocoClasses.names.count) ? CocoClasses.names[classIndex] : "Class \(classIndex)"
                
                boxes.append(BoundingBox(
                    rect: rect,
                    classIndex: classIndex,
                    score: score,
                    label: label
                ))
            }
        }
        // No NMS needed, model did it
        return boxes
    }

    private func postprocess(output: ZeticMLange.Tensor, originalSize: CGSize, classCount: Int, outputAnchors: Int) -> [BoundingBox] {
        let data: [Float] = output.data.withUnsafeBytes {
            let buffer = $0.bindMemory(to: Float.self)
            return Array(buffer)
        }
        
        var boxes: [BoundingBox] = []
        
        for anchorIndex in 0..<outputAnchors {
            var maxScore: Float = 0
            var maxClassIndex = -1
            
            for c in 0..<classCount {
                let classRow = 4 + c
                // Assuming [1, 4+C, Anchors] layout flattened: row * outputAnchors + col
                let index = classRow * outputAnchors + anchorIndex
                if index < data.count {
                    let score = data[index]
                    if score > maxScore {
                        maxScore = score
                        maxClassIndex = c
                    }
                }
            }
            
                if maxScore > confidenceThreshold {
                    // Debug block removed for performance

                
                let xc = data[0 * outputAnchors + anchorIndex]
                let yc = data[1 * outputAnchors + anchorIndex]
                let w = data[2 * outputAnchors + anchorIndex]
                let h = data[3 * outputAnchors + anchorIndex]
                
                let x = xc - w / 2
                let y = yc - h / 2
                
                let rect = CGRect(
                    x: CGFloat(x) / 640.0,
                    y: CGFloat(y) / 640.0,
                    width: CGFloat(w) / 640.0,
                    height: CGFloat(h) / 640.0
                )
                
                let label = (maxClassIndex >= 0 && maxClassIndex < CocoClasses.names.count) ? CocoClasses.names[maxClassIndex] : "Class \(maxClassIndex)"
                
                boxes.append(BoundingBox(
                    rect: rect,
                    classIndex: maxClassIndex,
                    score: maxScore,
                    label: label
                ))
            }
        }
        
        return nms(boxes: boxes)
    }
    
    private func nms(boxes: [BoundingBox]) -> [BoundingBox] {
        let sortedBoxes = boxes.sorted { $0.score > $1.score }
        var selected: [BoundingBox] = []
        
        for box in sortedBoxes {
            var keep = true
            for other in selected {
                if iou(boxA: box.rect, boxB: other.rect) > CGFloat(iouThreshold) {
                    keep = false
                    break
                }
            }
            if keep {
                selected.append(box)
            }
        }
        return selected
    }
    
    private func iou(boxA: CGRect, boxB: CGRect) -> CGFloat {
        let intersection = boxA.intersection(boxB)
        let interArea = intersection.width * intersection.height
        if intersection.width <= 0 || intersection.height <= 0 { return 0 }
        
        let boxAArea = boxA.width * boxA.height
        let boxBArea = boxB.width * boxB.height
        
        return interArea / (boxAArea + boxBArea - interArea)
    }
}
