import SwiftUI
import AVFoundation
import Vision

import ZeticMLange

struct YOLOv8View: View {
    @StateObject private var yolov8: YOLOv8 = YOLOv8()
    @StateObject private var cameraModel: CameraModel = CameraModel(.back, .frame)
    private let classes = CocoConfig.readClasses()
    @Environment(\.scenePhase) private var scenePhase
    @State private var aspectRatio: CGFloat = 16.0 / 9.0
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        let cameraResolution = cameraModel.resolution
        GeometryReader { geometry in
            ZStack {
                Color.black.edgesIgnoringSafeArea(.all)
                
                let width = geometry.size.width
                let height = geometry.size.height
                let targetWidth = width
                let targetHeight = targetWidth * aspectRatio
                
                CameraPreview(session: cameraModel.session)
                    .customFrame(targetWidth: targetWidth, targetHeight: targetHeight,
                                 geometryWidth: width, geometryHeight: height)
            }
            
            ZStack {
                // Unwrapping frame and timestamp into separate variables
                if let frame = cameraModel.frame, let timestamp = cameraModel.timestamp {
                    let input = YOLOv8Input(frame: frame, timestamp: timestamp)
                    let _ = yolov8.process(input: input)
                }
                
                ForEach(Array(zip(yolov8.detectionResults.indices, yolov8.detectionResults)), id: \.0) { index, result in
                    let color = getClassColor(classId: result.classId)
                    let confidence = String(format: "%.2f", result.confidence)
                    let label = "\(classes[Int(result.classId)]) \(confidence)"
                    
                    let width = geometry.size.width
                    let height = geometry.size.height
                    let targetWidth = width
                    let targetHeight = targetWidth * aspectRatio
                    
                    let box = calculateBox(for: result, in: CGSize(width: targetWidth, height: targetHeight), cameraResolution)
                    
                    DetectionBoxView(
                        result: result,
                        color: color,
                        label: label,
                        box: box)
                    .customFrame(targetWidth: targetWidth, targetHeight: targetHeight,
                                 geometryWidth: width, geometryHeight: height)
                }
            }
        }
        .ignoresSafeArea()
        .onAppear {
            cameraModel.checkCameraPermission()
        }
        .onDisappear {
            cameraModel.close()
            yolov8.close()
            dismiss()
        }
        .onChange(of: scenePhase) { newPhase in
            switch newPhase {
            case .background:
                cameraModel.close()
                yolov8.close()
                dismiss()
            case .inactive:
                break
            case .active:
                cameraModel.checkCameraPermission()
                break
            @unknown default:
                break
            }
        }
    }
    
    private func getClassColor(classId: Int32) -> Color {
        let r = Double((Int(classId) + 72) * 1717 % 256) / 255.0
        let g = Double((Int(classId) + 7) * 33 % 126 + 70) / 255.0
        let b = Double((Int(classId) + 47) * 107 % 256) / 255.0
        return Color(red: r, green: g, blue: b)
    }
    
    private func calculateBox(for result: YOLOv8Result, in targetSize: CGSize, _ cameraResolution: CGSize) -> [CGFloat] {
        let ret = [
            CGFloat(result.box[0]) * (targetSize.width / cameraResolution.height),
            CGFloat(result.box[1]) * (targetSize.height / cameraResolution.width),
            CGFloat(result.box[2]) * (targetSize.width / cameraResolution.height),
            CGFloat(result.box[3]) * (targetSize.height / cameraResolution.width)
        ]
        return ret
    }
    
}
