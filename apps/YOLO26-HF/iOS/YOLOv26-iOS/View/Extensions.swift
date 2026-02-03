import SwiftUI
import UIKit

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
