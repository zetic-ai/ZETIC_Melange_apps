import SwiftUI

/// Semicircular well-being gauge echoing Aiberry's results screen, drawn in the
/// app's palette. `value` is 0…1 (higher = brighter affect); the arc runs red
/// (low) on the left to green (bright) on the right with a marker at the value.
struct ScoreGauge: View {
    var value: Double          // 0…1
    var number: Int            // shown big (the 0–100 score)
    var band: String           // descriptive word, e.g. "Steady"

    private let lineWidth: CGFloat = 18
    private let height: CGFloat = 160
    private let colors: [Color] = [
        Color(red: 0.83, green: 0.36, blue: 0.33),   // red  (low)
        Color(red: 0.90, green: 0.62, blue: 0.26),   // orange
        Color(red: 0.86, green: 0.78, blue: 0.30),   // yellow
        Color(red: 0.40, green: 0.62, blue: 0.36),   // green (bright)
    ]

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let d = min(w, (height - lineWidth) * 2)          // diameter that fits
            let r = (d - lineWidth) / 2
            let center = CGPoint(x: w / 2, y: height - lineWidth / 2)
            let theta = Double.pi * (1 + min(max(value, 0), 1))   // π … 2π

            ZStack {
                Circle()
                    .trim(from: 0.5, to: 1.0)                      // top semicircle
                    .stroke(AngularGradient(gradient: Gradient(colors: colors),
                                            center: .center,
                                            startAngle: .degrees(180),
                                            endAngle: .degrees(360)),
                            style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))
                    .frame(width: d, height: d)
                    .position(center)

                Circle()
                    .fill(.white)
                    .frame(width: lineWidth + 6, height: lineWidth + 6)
                    .overlay(Circle().stroke(Theme.accent, lineWidth: 4))
                    .position(x: center.x + r * cos(theta), y: center.y + r * sin(theta))
                    .animation(.spring(response: 0.6, dampingFraction: 0.8), value: value)

                VStack(spacing: 0) {
                    Text("\(number)").font(.serif(46, .semibold)).foregroundStyle(Theme.ink)
                    Text(band).font(.serif(22)).foregroundStyle(Theme.inkSoft)
                }
                .position(x: center.x, y: center.y - d * 0.16)
            }
        }
        .frame(height: height)
        .frame(maxWidth: .infinity)
    }
}
