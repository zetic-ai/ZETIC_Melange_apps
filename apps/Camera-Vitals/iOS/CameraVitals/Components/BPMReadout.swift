import SwiftUI

/// Large heart-rate readout with a heart that pulses at the measured rate.
struct BPMReadout: View {
    let bpm: Double?
    let quality: Double

    @State private var beat = false

    private var bpmText: String {
        guard let bpm, bpm > 0 else { return "––" }
        return String(Int(bpm.rounded()))
    }

    private var beatInterval: Double {
        guard let bpm, bpm > 30 else { return 1 }
        return 60.0 / bpm
    }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 14) {
            Image(systemName: "heart.fill")
                .font(.system(size: 30))
                .foregroundStyle(Theme.accent)
                .scaleEffect(beat ? 1.18 : 0.92)
                .opacity(bpm == nil ? 0.35 : 1)
                .animation(.easeOut(duration: 0.18), value: beat)
                .onChange(of: beatInterval) { _ in restartBeat() }
                .onAppear { restartBeat() }

            Text(bpmText)
                .font(.system(size: 76, weight: .bold, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(Theme.textPrimary)
                .contentTransition(.numericText())
                .animation(.easeInOut(duration: 0.3), value: bpmText)

            Text("BPM")
                .font(.system(size: 20, weight: .semibold, design: .rounded))
                .foregroundStyle(Theme.textSecondary)
        }
    }

    private func restartBeat() {
        guard bpm != nil else { return }
        beat = false
        withAnimation(.easeOut(duration: 0.18).repeatForever(autoreverses: true).delay(0)) {
            beat = true
        }
    }
}
