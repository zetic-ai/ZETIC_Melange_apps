import SwiftUI

struct BoundingBoxView: View {
    let boxes: [BoundingBox]
    let geometry: GeometryProxy
    let sourceSize: CGSize
    let contentMode: ContentMode
    var isMirrored: Bool = false
    
    var body: some View {
        let viewW = geometry.size.width
        let viewH = geometry.size.height
        
        let videoRatio = sourceSize.width / sourceSize.height
        let viewRatio = viewW / viewH
        
        var renderW: CGFloat = viewW
        var renderH: CGFloat = viewH
        var offsetX: CGFloat = 0
        var offsetY: CGFloat = 0
        
        if contentMode == .fill {
            // Aspect Fill (Camera)
            if viewRatio > videoRatio {
                renderW = viewW
                renderH = viewW / videoRatio
                offsetY = (viewH - renderH) / 2
            } else {
                renderH = viewH
                renderW = viewH * videoRatio
                offsetX = (viewW - renderW) / 2
            }
        } else {
            // Aspect Fit (Image)
            if viewRatio > videoRatio {
                renderH = viewH
                renderW = renderH * videoRatio
                offsetX = (viewW - renderW) / 2
            } else {
                renderW = viewW
                renderH = renderW / videoRatio
                offsetY = (viewH - renderH) / 2
            }
        }
        
        return ZStack(alignment: .topLeading) {
            ForEach(boxes) { box in
                let rect = box.rect
                
                // Map normalized coordinates to the Rendered Video Size
                let normalizedX = isMirrored ? (1.0 - rect.maxX) : rect.minX
                
                let x = normalizedX * renderW + offsetX
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
    }
}
