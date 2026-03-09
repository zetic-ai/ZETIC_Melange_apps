import SwiftUI

enum AppFont {
    static func title(_ size: CGFloat = 28, weight: Font.Weight = .semibold) -> Font { .system(size: size, weight: weight, design: .rounded) }
    static func body(_ size: CGFloat = 17, weight: Font.Weight = .regular) -> Font { .system(size: size, weight: weight, design: .rounded) }
    static func mono(_ size: CGFloat = 14) -> Font { .system(size: size, weight: .regular, design: .monospaced) }
}
