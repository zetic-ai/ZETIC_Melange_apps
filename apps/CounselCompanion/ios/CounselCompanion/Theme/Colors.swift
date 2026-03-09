import SwiftUI

extension Color {
    static let bgTop = Color("BgTop")
    static let bgBottom = Color("BgBottom")

    // Warm palette
    static let warmSage = Color(red: 0.35, green: 0.49, blue: 0.35)
    static let warmSageLight = Color(red: 0.69, green: 0.83, blue: 0.66)
    static let warmCream = Color(red: 0.98, green: 0.96, blue: 0.94)
    static let warmLinen = Color(red: 0.99, green: 0.97, blue: 0.95)
    static let warmPeach = Color(red: 0.83, green: 0.52, blue: 0.42)
    static let warmPeachLight = Color(red: 0.96, green: 0.77, blue: 0.69)
}

extension ShapeStyle where Self == Color {
    static var bubbleUser: Color {
        Color(red: 0.35, green: 0.49, blue: 0.35)
    }
    static var bubbleUserEnd: Color {
        Color(red: 0.35, green: 0.49, blue: 0.35).opacity(0.85)
    }
}
