import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var models: AppModels

    var body: some View {
        TabView {
            NavigationStack { EmotionView(model: models.emotion) }
                .tabItem { Label("Emotion", systemImage: "waveform") }

            NavigationStack { RespiratoryView(model: models.yamnet) }
                .tabItem { Label("Respiratory", systemImage: "lungs.fill") }

            NavigationStack { AboutView() }
                .tabItem { Label("About", systemImage: "info.circle") }
        }
        .tint(Theme.tileInk)
    }
}

/// The pitch, in-app: each tab is a model running on the same on-device pipeline.
/// Swapping in a client's model is a one-line change.
struct AboutView: View {
    private let models: [(String, String, String)] = [
        ("waveform", "Emotion", "wav2vec2-large-xlsr · 7-class speech emotion"),
        ("lungs.fill", "Respiratory", "google/Sound Classification (YAMNET)")
    ]

    var body: some View {
        ScrollView {
            VStack(spacing: 18) {
                EditorialTitle(leading: "Voice Biomarkers", emphasis: "")
                    .padding(.top, 8).padding(.bottom, 2)

                VStack(alignment: .leading, spacing: 8) {
                    Text("Every model runs fully on this phone through ZETIC Melange, accelerated on the Apple Neural Engine. Microphone audio is never uploaded — it works in Airplane Mode.")
                        .font(.subheadline).foregroundStyle(Theme.inkSoft)
                }
                .card()

                VStack(alignment: .leading, spacing: 16) {
                    Text("Two models, one pipeline").font(.headline).foregroundStyle(Theme.ink)
                    ForEach(models, id: \.1) { item in
                        HStack(spacing: 14) {
                            IconTile(system: item.0, size: 48)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(item.1).font(.subheadline.weight(.semibold)).foregroundStyle(Theme.ink)
                                Text(item.2).font(.caption).foregroundStyle(Theme.inkSoft)
                            }
                            Spacer(minLength: 0)
                        }
                    }
                }
                .card()

                VStack(alignment: .leading, spacing: 12) {
                    Label("Why on-device", systemImage: "lock.shield.fill")
                        .font(.headline).foregroundStyle(Theme.tileInk)
                    bullet("HIPAA-friendly: sensitive audio never leaves the device")
                    bullet("Real-time: NPU inference in milliseconds")
                    bullet("Offline: no connectivity required")
                    bullet("No cloud-GPU cost per inference")
                }
                .card()

                VStack(alignment: .leading, spacing: 8) {
                    Label("Deploy your own model", systemImage: "arrow.triangle.2.circlepath")
                        .font(.headline).foregroundStyle(Theme.ink)
                    Text("Upload your model to mlange.zetic.ai, then change one line:")
                        .font(.caption).foregroundStyle(Theme.inkSoft)
                    Text("ZeticMLangeModel(personalKey:, name: \"your-org/your-model\")")
                        .font(.caption2.monospaced()).foregroundStyle(Theme.ink)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Theme.cardAlt)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                .card()
            }
            .padding(20)
        }
        .background(Theme.bg.ignoresSafeArea())
        .toolbar(.hidden, for: .navigationBar)
    }

    private func bullet(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.tileInk).font(.caption)
            Text(text).font(.subheadline).foregroundStyle(Theme.ink)
        }
    }
}
