import SwiftUI

/// A conversation bubble — the app's question (leading, light card) or the user's
/// answer (trailing, deep-green), in the shared VoiceVitals palette.
struct ChatBubble: View {
    enum Role { case prompt, user }
    let role: Role
    let text: String

    var body: some View {
        HStack {
            if role == .user { Spacer(minLength: 40) }
            Text(text)
                .font(role == .prompt ? .body : .body.weight(.medium))
                .foregroundStyle(role == .prompt ? Theme.ink : .white)
                .padding(.horizontal, 16).padding(.vertical, 12)
                .background(role == .prompt ? Theme.cardAlt : Theme.tileInk)
                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            if role == .prompt { Spacer(minLength: 40) }
        }
    }
}

/// Animated "listening…" placeholder shown in the user's bubble while recording.
struct ListeningBubble: View {
    @State private var phase = 0
    private let timer = Timer.publish(every: 0.4, on: .main, in: .common).autoconnect()

    var body: some View {
        HStack {
            Spacer(minLength: 40)
            HStack(spacing: 5) {
                ForEach(0..<3) { i in
                    Circle().fill(.white.opacity(phase == i ? 1 : 0.4))
                        .frame(width: 7, height: 7)
                }
            }
            .padding(.horizontal, 18).padding(.vertical, 14)
            .background(Theme.tileInk.opacity(0.9))
            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        }
        .onReceive(timer) { _ in phase = (phase + 1) % 3 }
    }
}
