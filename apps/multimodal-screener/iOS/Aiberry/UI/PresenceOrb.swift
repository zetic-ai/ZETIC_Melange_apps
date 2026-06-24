import SwiftUI

/// A calm, professional "presence" indicator in the shared sage palette — the focal
/// element during the guided check-in. No cartoon face, no emoji: it mirrors
/// VoiceVitals' RecordButton / IconTile look. A live input-level ring shows it's
/// listening; a quiet spinner shows it's working.
struct PresenceOrb: View {
    var listening: Bool = false
    var level: Float = 0
    var thinking: Bool = false
    var size: CGFloat = 170

    var body: some View {
        ZStack {
            Circle()
                .stroke(Theme.tileInk.opacity(0.28), lineWidth: 2 + CGFloat(level) * 14)
                .frame(width: size * 1.12, height: size * 1.12)
                .opacity(listening ? 1 : 0)
                .animation(.easeOut(duration: 0.12), value: level)

            Circle()
                .fill(Theme.tile)
                .frame(width: size, height: size)
                .overlay(Circle().stroke(Theme.tileInk.opacity(0.15), lineWidth: 1))

            if thinking {
                ProgressView().tint(Theme.tileInk).scaleEffect(1.4)
            } else {
                Image(systemName: listening ? "waveform" : "person.fill")
                    .font(.system(size: size * 0.32, weight: .light))
                    .foregroundStyle(Theme.tileInk)
            }
        }
        .frame(width: size * 1.12, height: size * 1.12)
    }
}
