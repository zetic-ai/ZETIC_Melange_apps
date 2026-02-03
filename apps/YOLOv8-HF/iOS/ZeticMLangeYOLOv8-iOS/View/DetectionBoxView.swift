import SwiftUI

import ZeticMLange

struct DetectionBoxView: View {
    let result: YOLOv8Result
    let color: Color
    let label: String
    let box: [CGFloat]
    
    var body: some View {
        ZStack {
            // Bounding box
            Path { path in
                let rect = CGRect(
                    x: box[0],
                    y: box[1],
                    width: box[2],
                    height: box[3]
                )
                path.addRect(rect)
            }
            .stroke(color, lineWidth: 5)
            
            // Label background
            Path { path in
                let labelHeight: CGFloat = 25
                let labelWidth = CGFloat(label.count) * 15
                let rect = CGRect(
                    x: box[0],
                    y: box[1] - labelHeight,
                    width: labelWidth,
                    height: labelHeight
                )
                path.addRect(rect)
            }
            .fill(color)
            
            // Label text
            Text(label)
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(.black)
                .position(
                    x: box[0] + CGFloat(label.count) * 7.5,
                    y: box[1] - 12.5
                )
        }
    }
}
