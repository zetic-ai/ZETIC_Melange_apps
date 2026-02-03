import SwiftUI

struct CustomFrameModifier: ViewModifier {
    let targetWidth: CGFloat
    let targetHeight: CGFloat
    let geometryWidth: CGFloat
    let geometryHeight: CGFloat
    
    func body(content: Content) -> some View {
        content
            .frame(width: targetWidth, height: targetHeight)
            .position(x: geometryWidth / 2, y: geometryHeight / 2)
            .edgesIgnoringSafeArea(.all)
    }
}

extension View {
    func customFrame(targetWidth: CGFloat, targetHeight: CGFloat, geometryWidth: CGFloat, geometryHeight: CGFloat) -> some View {
        self.modifier(CustomFrameModifier(targetWidth: targetWidth, targetHeight: targetHeight,
                                          geometryWidth: geometryWidth, geometryHeight: geometryHeight))
    }
}
