import SwiftUI

/// Ranked probability bars for the full 7-class softmax distribution. Each row is a
/// class name, a glowing fill proportional to probability, and a mono percentage.
struct ClassDistributionBars: View {
    let scores: [ClassScore]
    var maxRows: Int = 5

    @State private var appeared = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Full distribution")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(Theme.inkFaint)
                .textCase(.uppercase)
                .tracking(1.4)

            ForEach(Array(scores.prefix(maxRows).enumerated()), id: \.element.id) { index, score in
                row(score, isTop: index == 0)
            }
        }
        .onAppear { withAnimation(.easeOut(duration: 0.7)) { appeared = true } }
    }

    private func row(_ score: ClassScore, isTop: Bool) -> some View {
        VStack(spacing: 5) {
            HStack {
                Text(score.skinClass.title)
                    .font(.system(size: 13, weight: isTop ? .semibold : .regular))
                    .foregroundStyle(isTop ? Theme.ink : Theme.inkSoft)
                Spacer()
                Text("\(Int((score.probability * 100).rounded()))%")
                    .font(Theme.mono(12, isTop ? .semibold : .regular))
                    .foregroundStyle(isTop ? score.skinClass.tint : Theme.inkFaint)
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.06))
                    Capsule()
                        .fill(LinearGradient(
                            colors: [score.skinClass.tint.opacity(0.6), score.skinClass.tint],
                            startPoint: .leading, endPoint: .trailing))
                        .frame(width: appeared ? max(4, geo.size.width * CGFloat(score.probability)) : 0)
                        .shadow(color: score.skinClass.tint.opacity(isTop ? 0.5 : 0), radius: 5)
                }
            }
            .frame(height: 7)
        }
    }
}
