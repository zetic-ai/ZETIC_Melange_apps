import SwiftUI

/// Circular confidence gauge: a tinted ring that sweeps to `value` with the
/// percentage shown in a monospaced readout at center.
struct ConfidenceRing: View {
    let value: Float          // 0…1
    var tint: Color = Theme.accent
    var size: CGFloat = 132

    @State private var animated: CGFloat = 0

    var body: some View {
        ZStack {
            Circle()
                .stroke(Color.white.opacity(0.08), lineWidth: 10)

            Circle()
                .trim(from: 0, to: animated)
                .stroke(
                    AngularGradient(colors: [tint.opacity(0.7), tint], center: .center),
                    style: StrokeStyle(lineWidth: 10, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))
                .shadow(color: tint.opacity(0.5), radius: 8)

            VStack(spacing: 2) {
                Text("\(Int((value * 100).rounded()))")
                    .font(Theme.mono(34, .semibold))
                    .foregroundColor(Theme.ink)
                + Text("%")
                    .font(Theme.mono(18, .medium))
                    .foregroundColor(Theme.inkSoft)
                Text("confidence")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Theme.inkFaint)
                    .textCase(.uppercase)
                    .tracking(1.2)
            }
        }
        .frame(width: size, height: size)
        .onAppear { animate() }
        .onChange(of: value) { _ in animate() }
    }

    private func animate() {
        animated = 0
        withAnimation(.easeOut(duration: 0.9)) { animated = CGFloat(value) }
    }
}
