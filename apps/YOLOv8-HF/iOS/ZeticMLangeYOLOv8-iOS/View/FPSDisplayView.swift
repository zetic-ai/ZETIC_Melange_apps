import SwiftUI

struct FPSDisplayView: View {
    let fps: Double
    let modelKey: String
    
    var body: some View {
        VStack {
            Spacer()
            HStack(alignment: .bottom) {
                Spacer()
                Text(String(format: "FPS: %.1f\nModel: %@", fps, modelKey))
                    .font(.system(size: 20, weight: .bold))
                    .foregroundColor(.white)
                    .padding(8)
                    .background(Color.black.opacity(0.6))
                    .cornerRadius(8)
                    .padding()
            }
        }
    }
}
