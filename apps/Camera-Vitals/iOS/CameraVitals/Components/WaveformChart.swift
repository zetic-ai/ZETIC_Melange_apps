import SwiftUI

/// Scrolling PPG waveform drawn with Canvas, auto-scaled to its min/max.
struct WaveformChart: View {
    let samples: [Float]
    var color: Color = Theme.accent

    var body: some View {
        Canvas { context, size in
            guard samples.count > 1 else { return }
            let lo = samples.min() ?? -1
            let hi = samples.max() ?? 1
            let range = max(hi - lo, 1e-4)

            var path = Path()
            let stepX = size.width / CGFloat(samples.count - 1)
            for (i, v) in samples.enumerated() {
                let x = CGFloat(i) * stepX
                let norm = CGFloat((v - lo) / range)              // 0...1
                let y = size.height * (1 - norm) * 0.8 + size.height * 0.1
                if i == 0 { path.move(to: CGPoint(x: x, y: y)) }
                else { path.addLine(to: CGPoint(x: x, y: y)) }
            }
            context.stroke(
                path,
                with: .color(color),
                style: StrokeStyle(lineWidth: 2.5, lineCap: .round, lineJoin: .round)
            )
        }
    }
}
