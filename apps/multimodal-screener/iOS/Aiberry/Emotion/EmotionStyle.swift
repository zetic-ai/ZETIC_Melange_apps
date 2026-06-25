import SwiftUI

/// Visual identity (emoji + color) for each emotion label, shared by the voice
/// model, the face model, and the fused readout.
enum EmotionStyle {
    static func emoji(_ label: String) -> String {
        switch label {
        case "Angry":    return "😠"
        case "Disgust":  return "🤢"
        case "Fear":     return "😨"
        case "Happy":    return "😄"
        case "Neutral":  return "😐"
        case "Sad":      return "😢"
        case "Surprise": return "😲"
        default:          return "🫥"
        }
    }

    static func color(_ label: String) -> Color {
        switch label {
        case "Angry":    return Color(red: 0.90, green: 0.30, blue: 0.30)
        case "Disgust":  return Color(red: 0.45, green: 0.65, blue: 0.30)
        case "Fear":     return Color(red: 0.55, green: 0.40, blue: 0.85)
        case "Happy":    return Color(red: 0.97, green: 0.74, blue: 0.20)
        case "Neutral":  return Color(red: 0.55, green: 0.58, blue: 0.62)
        case "Sad":      return Color(red: 0.30, green: 0.55, blue: 0.90)
        case "Surprise": return Color(red: 0.95, green: 0.55, blue: 0.25)
        default:          return Theme.accent
        }
    }
}

/// A single (label, probability) pair, used across voice / face / fused results.
struct EmotionScore: Identifiable, Equatable {
    let id = UUID()
    let label: String
    let probability: Float
}
