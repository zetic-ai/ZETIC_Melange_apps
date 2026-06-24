import SwiftUI

struct EmotionView: View {
    @StateObject private var recorder = AudioRecorder()
    @StateObject private var player = SamplePlayer()
    @ObservedObject var model: EmotionModel
    @State private var showResults = false

    var body: some View {
        ScrollView {
            VStack(spacing: 18) {
                EditorialTitle(leading: "Speech Emotions", emphasis: "")
                    .padding(.top, 8).padding(.bottom, 2)

                PrivacyBanner()

                VStack(spacing: 6) {
                    CardHeader(icon: "waveform",
                               title: "Speech Emotion",
                               subtitle: "wav2vec2-large-xlsr · 7 emotions")
                }
                .card()

                recorderCard

                samplesCard
            }
            .padding(20)
        }
        .background(Theme.bg.ignoresSafeArea())
        .toolbar(.hidden, for: .navigationBar)
        .onChange(of: model.status) { newStatus in
            if newStatus == .ready, model.top != nil { showResults = true }
        }
        .sheet(isPresented: $showResults) { resultsSheet }
    }

    private var resultsSheet: some View {
        ScrollView {
            VStack(spacing: 18) {
                if let top = model.top {
                    heroCard(top: top)
                    breakdownCard(top: top)
                }
            }
            .padding(20)
            .animation(.spring(response: 0.45, dampingFraction: 0.85), value: model.scores)
        }
        .background(Theme.bg.ignoresSafeArea())
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
    }

    private var recorderCard: some View {
        VStack(spacing: 14) {
            RecordButton(isRecording: recorder.isRecording,
                         level: recorder.level,
                         busy: model.status.isBusy) { toggle() }
            Text(recorder.isRecording ? "Recording… tap to stop" : "Tap to record")
                .font(.subheadline).foregroundStyle(Theme.inkSoft)
            HStack { StatusLine(status: model.status); Spacer(); LatencyBadge(ms: model.latencyMs) }
        }
        .card()
    }

    /// Plain audio players. Start a recording, play a sample so the mic hears it,
    /// then stop the recording to run the model on what was captured.
    private var samplesCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Audio samples").font(.subheadline.weight(.semibold)).foregroundStyle(Theme.inkSoft)
            HStack(spacing: 12) {
                sampleButton(label: "Sample 1", resource: "angry-voice")
                sampleButton(label: "Sample 2", resource: "happy-voice")
            }
        }
        .card()
    }

    private func sampleButton(label: String, resource: String) -> some View {
        let isActive = player.playing == resource
        return Button { playSample(resource: resource) } label: {
            HStack(spacing: 8) {
                Image(systemName: isActive ? "stop.fill" : "play.fill")
                Text(label).font(.subheadline.weight(.semibold))
            }
            .foregroundStyle(Theme.tileInk)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(Theme.tile.opacity(0.5))
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(isActive ? "Stop \(label)" : "Play \(label)")
    }

    private func playSample(resource: String) {
        Haptics.tap()
        if player.playing == resource {
            player.stop()
        } else {
            player.play(resource: resource, ext: "mp3", key: resource)
        }
    }

    private func heroCard(top: EmotionScore) -> some View {
        let tint = EmotionStyle.color(top.label)
        return VStack(spacing: 8) {
            Text(EmotionStyle.emoji(top.label)).font(.system(size: 60))
            Text(top.label).font(.serif(40)).foregroundStyle(Theme.ink)
            Text("\(Int((top.probability * 100).rounded()))% confidence")
                .font(.subheadline).foregroundStyle(Theme.inkSoft)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 26)
        .background(
            LinearGradient(colors: [tint.opacity(0.30), tint.opacity(0.08)],
                           startPoint: .top, endPoint: .bottom)
        )
        .clipShape(RoundedRectangle(cornerRadius: Theme.corner, style: .continuous))
        .id(top.label)
        .transition(.scale.combined(with: .opacity))
    }

    private func breakdownCard(top: EmotionScore) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("All emotions").font(.subheadline.weight(.semibold)).foregroundStyle(Theme.inkSoft)
            ForEach(model.scores) { score in
                BarRow(label: "\(EmotionStyle.emoji(score.label))  \(score.label)",
                       value: score.probability,
                       tint: EmotionStyle.color(score.label),
                       highlighted: score.id == top.id)
            }
        }
        .card()
    }

    private func toggle() {
        if recorder.isRecording {
            player.stop()              // finish the sample clip along with the recording
            recorder.stop()
        } else {
            recorder.record(autoStopSeconds: nil) { samples in model.analyze(samples) }
        }
    }
}
