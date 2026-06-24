import SwiftUI

struct RespiratoryView: View {
    @StateObject private var recorder = AudioRecorder()
    @StateObject private var player = SamplePlayer()
    @ObservedObject var model: YamnetModel
    @State private var showResults = false

    var body: some View {
        ScrollView {
            VStack(spacing: 18) {
                EditorialTitle(leading: "Respiratory Sounds", emphasis: "")
                    .padding(.top, 8).padding(.bottom, 2)

                PrivacyBanner()

                VStack(spacing: 6) {
                    CardHeader(icon: "lungs.fill",
                               title: "Acoustic Events",
                               subtitle: "YAMNet · cough / breath / wheeze")
                }
                .card()

                VStack(spacing: 14) {
                    RecordButton(isRecording: recorder.isRecording,
                                 level: recorder.level,
                                 busy: model.status.isBusy) {
                        recorder.record(autoStopSeconds: 3.0) { samples in model.analyze(samples) }
                    }
                    Text(recorder.isRecording ? "Listening… cough or breathe" : "Tap, then cough or breathe")
                        .font(.subheadline).foregroundStyle(Theme.inkSoft)
                    HStack { StatusLine(status: model.status); Spacer(); LatencyBadge(ms: model.latencyMs) }
                }
                .card()

                samplesCard
            }
            .padding(20)
        }
        .background(Theme.bg.ignoresSafeArea())
        .toolbar(.hidden, for: .navigationBar)
        .onChange(of: model.status) { newStatus in
            if newStatus == .ready, model.topRespiratory != nil { showResults = true }
        }
        .sheet(isPresented: $showResults) { resultsSheet }
    }

    private var resultsSheet: some View {
        ScrollView {
            VStack(spacing: 18) {
                if let top = model.topRespiratory {
                    VStack(alignment: .leading, spacing: 14) {
                        HStack {
                            Text("Top respiratory event").font(.subheadline).foregroundStyle(Theme.inkSoft)
                            Spacer()
                            Text(top.name).font(.serif(22)).foregroundStyle(Theme.ink)
                        }
                        Divider()
                        ForEach(model.respiratoryEvents.prefix(6)) { event in
                            BarRow(label: event.name,
                                   value: event.score,
                                   tint: Theme.tileInk,
                                   highlighted: event.id == top.id)
                        }
                    }
                    .card()
                }

                if !model.topEvents.isEmpty {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("All sounds (top 5)").font(.subheadline.weight(.semibold)).foregroundStyle(Theme.inkSoft)
                        ForEach(model.topEvents) { event in
                            HStack {
                                Text(event.name).font(.caption).foregroundStyle(Theme.ink)
                                Spacer()
                                Text("\(Int((event.score*100).rounded()))%")
                                    .font(.caption.monospacedDigit()).foregroundStyle(Theme.inkSoft)
                            }
                        }
                    }
                    .card()
                }
            }
            .padding(20)
            .animation(.default, value: model.respiratoryEvents)
        }
        .background(Theme.bg.ignoresSafeArea())
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
    }

    /// Plain audio players. Start a recording, play a sample so the mic hears it,
    /// then let the recording finish to run the model on what was captured.
    private var samplesCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Audio samples").font(.subheadline.weight(.semibold)).foregroundStyle(Theme.inkSoft)
            HStack(spacing: 12) {
                sampleButton(label: "Sample 1", resource: "cough-sound")
                sampleButton(label: "Sample 2", resource: "sigh-sound")
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
}
