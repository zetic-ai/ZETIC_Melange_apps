import SwiftUI

/// End-of-session "Screening Insights" — mirrors Aiberry's results screen (score
/// gauge + tabbed insights + transcript), rendered in the app's design language.
struct InsightsView: View {
    let report: ScreeningReport
    let onRestart: () -> Void

    enum Tab: String, CaseIterable { case score = "Score", insights = "Screening Insights", transcript = "Transcript" }
    @State private var tab: Tab = .score

    var body: some View {
        ScrollView {
            VStack(spacing: 18) {
                header
                tabBar
                switch tab {
                case .score:      scoreTab
                case .insights:   insightsTab
                case .transcript: transcriptTab
                }
                DisclaimerNote().padding(.top, 4)
                PrimaryButton(title: "New check-in", enabled: true) { onRestart() }
            }
            .padding(20)
        }
    }

    private var header: some View {
        HStack {
            Text("Results").font(.serif(30, .semibold)).foregroundStyle(Theme.ink)
            Spacer()
        }
    }

    private var tabBar: some View {
        HStack(spacing: 0) {
            ForEach(Tab.allCases, id: \.self) { t in
                VStack(spacing: 6) {
                    Text(t.rawValue)
                        .font(.caption.weight(tab == t ? .bold : .regular))
                        .foregroundStyle(tab == t ? Theme.accent : Theme.inkSoft)
                        .lineLimit(1).minimumScaleFactor(0.8)
                    Rectangle().fill(tab == t ? Theme.accent : .clear).frame(height: 2)
                }
                .frame(maxWidth: .infinity)
                .contentShape(Rectangle())
                .onTapGesture { withAnimation { tab = t } }
            }
        }
    }

    // MARK: Score

    private var scoreTab: some View {
        VStack(spacing: 16) {
            VStack(spacing: 8) {
                ScoreGauge(value: Double(report.wellbeing) / 100,
                           number: report.wellbeing, band: report.band)
                Text("Composite well-being")
                    .font(.caption).foregroundStyle(Theme.inkSoft)
            }
            .card()

            HStack(spacing: 12) {
                modalityChip(title: "Face", score: report.faceTop)
                modalityChip(title: "Voice", score: report.voiceTop)
            }

            HStack {
                Label("Evidence", systemImage: "checkmark.seal")
                    .font(.caption).foregroundStyle(Theme.inkSoft)
                Spacer()
                Text("\(Int((report.confidence * 100).rounded()))% · \(report.faceFrames) face reads")
                    .font(.caption.monospacedDigit()).foregroundStyle(Theme.inkSoft)
            }
            .card(14)
        }
    }

    private func modalityChip(title: String, score: EmotionScore?) -> some View {
        VStack(spacing: 6) {
            Text(title).font(.caption.weight(.semibold)).foregroundStyle(Theme.inkSoft)
            Text(score.map { EmotionStyle.emoji($0.label) } ?? "—").font(.system(size: 34))
            Text(score?.label ?? "n/a").font(.subheadline.weight(.medium)).foregroundStyle(Theme.ink)
        }
        .frame(maxWidth: .infinity).padding(.vertical, 14)
        .background(Theme.cardAlt)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    // MARK: Screening insights

    private var insightsTab: some View {
        VStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 16) {
                Text("Sub-dimensions").font(.subheadline.weight(.semibold)).foregroundStyle(Theme.inkSoft)
                dimensionRow("Mood", report.mood, "Mood")
                dimensionRow("Energy", report.energy, "Energy")
                dimensionRow("Rate of speech", report.rateOfSpeech, nil)
            }
            .card()

            VStack(alignment: .leading, spacing: 14) {
                Text("Emotion blend (face + voice)")
                    .font(.subheadline.weight(.semibold)).foregroundStyle(Theme.inkSoft)
                ForEach(report.fused) { score in
                    BarRow(label: "\(EmotionStyle.emoji(score.label))  \(score.label)",
                           value: score.probability,
                           tint: EmotionStyle.color(score.label),
                           highlighted: score.id == report.fused.first?.id)
                }
            }
            .card()
        }
    }

    private func dimensionRow(_ name: String, _ value: Int, _ driverKey: String?) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            BarRow(label: name, value: Float(value) / 100, tint: Theme.accent)
            if let key = driverKey, let drivers = report.drivers[key], !drivers.isEmpty {
                Text("Driven mostly by \(drivers.joined(separator: ", "))")
                    .font(.caption2).foregroundStyle(Theme.inkSoft)
            }
        }
    }

    // MARK: Transcript

    private var transcriptTab: some View {
        VStack(spacing: 12) {
            if report.transcript.allSatisfy({ $0.answer.isEmpty }) {
                VStack(spacing: 8) {
                    Image(systemName: "text.bubble").font(.title).foregroundStyle(Theme.inkSoft)
                    Text("No transcript available")
                        .font(.subheadline).foregroundStyle(Theme.ink)
                    Text("On-device speech recognition may be unavailable on this device or locale.")
                        .font(.caption).foregroundStyle(Theme.inkSoft).multilineTextAlignment(.center)
                }
                .card()
            } else {
                ForEach(report.transcript) { qa in
                    VStack(spacing: 8) {
                        ChatBubble(role: .prompt, text: qa.question)
                        if !qa.answer.isEmpty { ChatBubble(role: .user, text: qa.answer) }
                    }
                }
                HStack(spacing: 6) {
                    Image(systemName: "lock.fill").font(.caption2)
                    Text("Transcribed on-device").font(.caption2)
                }
                .foregroundStyle(Theme.inkSoft)
            }
        }
    }
}
