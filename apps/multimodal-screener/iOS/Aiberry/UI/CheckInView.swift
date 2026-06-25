import SwiftUI

/// Root of the guided check-in flow. Switches on the session phase between the
/// landing/consent screens, the conversation, the analyzing state, and results.
struct CheckInView: View {
    @ObservedObject var models: AppModels
    @ObservedObject var session: CheckInSession

    var body: some View {
        ZStack {
            Theme.bg.ignoresSafeArea()
            switch session.phase {
            case .idle:                 LandingView(models: models, session: session)
            case .intro:                IntroView(session: session)
            case .question:             ConversationView(session: session)
            case .analyzing:            AnalyzingView()
            case .insights(let report): InsightsView(report: report) { session.restart() }
            }
        }
        .animation(.easeInOut(duration: 0.25), value: session.phase)
    }
}

// MARK: - Small neutral header mark

/// A restrained header mark (sage square + serif label) — no brand colors.
struct BrandMark: View {
    var size: CGFloat = 20
    var body: some View {
        HStack(spacing: 8) {
            RoundedRectangle(cornerRadius: 5, style: .continuous)
                .fill(Theme.tile)
                .frame(width: size, height: size)
            Text("Screening").font(.serif(size, .semibold)).foregroundStyle(Theme.ink)
        }
    }
}

// MARK: - Landing

private struct LandingView: View {
    @ObservedObject var models: AppModels
    @ObservedObject var session: CheckInSession
    @ObservedObject private var voice: EmotionModel
    @ObservedObject private var face: FaceEmotionModel

    init(models: AppModels, session: CheckInSession) {
        self.models = models
        self.session = session
        self.voice = models.voice
        self.face = models.face
    }

    private var ready: Bool { models.bothReady }

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                EditorialTitle(leading: "Conversational", emphasis: "multimodal screener")
                    .padding(.top, 8)

                PrivacyBanner()

                VStack(spacing: 10) {
                    modelRow(title: "Facial expression", subtitle: "On-device · Melange", status: face.status)
                    Divider()
                    modelRow(title: "Voice tone", subtitle: "On-device · Melange", status: voice.status)
                }
                .card()

                PrimaryButton(title: ready ? "Start check-in" : "Preparing models…",
                              enabled: ready) {
                    session.showIntro()
                }
            }
            .padding(20)
        }
    }

    private func modelRow(title: String, subtitle: String, status: ModelStatus) -> some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.subheadline.weight(.semibold)).foregroundStyle(Theme.ink)
                Text(subtitle).font(.caption).foregroundStyle(Theme.inkSoft)
            }
            Spacer()
            StatusLine(status: status)
        }
    }
}

// MARK: - Intro / consent

private struct IntroView: View {
    @ObservedObject var session: CheckInSession

    var body: some View {
        VStack(spacing: 22) {
            Spacer()
            PresenceOrb(size: 150)
            Text("How it works")
                .font(.serif(30, .semibold)).foregroundStyle(Theme.ink)
            VStack(alignment: .leading, spacing: 14) {
                point("text.bubble", "Answer a few open questions out loud, naturally.")
                point("faceid", "Your expression and voice are analyzed on-device as you talk.")
                point("lock.shield.fill", "Nothing is recorded to the cloud. It works in Airplane Mode.")
            }
            .card()
            DisclaimerNote()
            Spacer()
            PrimaryButton(title: "Begin", enabled: true) { session.begin() }
        }
        .padding(20)
    }

    private func point(_ icon: String, _ text: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            IconTile(system: icon, size: 40)
            Text(text).font(.subheadline).foregroundStyle(Theme.ink)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

// MARK: - Conversation

private struct ConversationView: View {
    @ObservedObject var session: CheckInSession

    var body: some View {
        VStack(spacing: 0) {
            header

            Spacer(minLength: 16)

            VStack(spacing: 28) {
                PresenceOrb(listening: session.isRecording,
                            level: session.micLevel,
                            size: 168)
                VStack(spacing: 12) {
                    ChatBubble(role: .prompt, text: session.spokenText)
                    if session.isRecording { ListeningBubble() }
                }
            }

            Spacer(minLength: 16)

            nextButton.padding(.bottom, 8)
        }
        .padding(.horizontal, 20)
        .safeAreaInset(edge: .bottom) { bottomBar }
    }

    private var header: some View {
        HStack(alignment: .top) {
            BrandMark()
            Spacer()
            cameraPiP
        }
        .padding(.top, 8)
    }

    private var cameraPiP: some View {
        ZStack {
            if session.camera.authorized {
                CameraPreviewView(controller: session.camera)
            } else {
                Theme.dark.overlay(Image(systemName: "person.fill")
                    .font(.title).foregroundStyle(.white.opacity(0.4)))
            }
        }
        .frame(width: 96, height: 128)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Theme.tileInk.opacity(0.25), lineWidth: 1))
    }

    private var nextButton: some View {
        PrimaryButton(title: session.questionIndex + 1 < session.totalQuestions ? "Next question" : "See results",
                      systemImage: "arrow.right",
                      enabled: session.canAdvance) {
            session.advance()
        }
    }

    private var bottomBar: some View {
        HStack {
            Label("Mic", systemImage: session.isRecording ? "mic.fill" : "mic.slash.fill")
                .font(.caption.weight(.semibold))
                .foregroundStyle(session.isRecording ? Theme.accent : Theme.inkSoft)
                .frame(maxWidth: .infinity)

            VStack(spacing: 1) {
                Text("\(session.questionIndex + 1)/\(session.totalQuestions)")
                    .font(.subheadline.weight(.bold)).foregroundStyle(Theme.ink)
                Text("Questions").font(.caption2).foregroundStyle(Theme.inkSoft)
            }
            .frame(maxWidth: .infinity)

            Button { Haptics.tap(); session.cancel() } label: {
                Label("End", systemImage: "xmark.circle.fill")
                    .font(.caption.weight(.semibold)).foregroundStyle(Theme.danger)
            }
            .buttonStyle(.plain)
            .frame(maxWidth: .infinity)
        }
        .padding(.vertical, 10)
        .background(Theme.bg)
    }
}

// MARK: - Analyzing

private struct AnalyzingView: View {
    var body: some View {
        VStack(spacing: 20) {
            PresenceOrb(thinking: true, size: 150)
            Text("Reading your check-in…")
                .font(.serif(24)).foregroundStyle(Theme.ink)
            Text("Combining face and voice, on-device")
                .font(.subheadline).foregroundStyle(Theme.inkSoft)
        }
        .padding(30)
    }
}

// MARK: - Shared

/// Primary filled action button in the deep-green accent.
struct PrimaryButton: View {
    let title: String
    var systemImage: String? = nil
    var enabled: Bool = true
    let action: () -> Void

    var body: some View {
        Button { Haptics.tap(); action() } label: {
            HStack(spacing: 8) {
                Text(title)
                if let systemImage { Image(systemName: systemImage) }
            }
            .font(.headline).foregroundStyle(.white)
            .frame(maxWidth: .infinity).padding(.vertical, 15)
            .background(enabled ? Theme.accent : Theme.accent.opacity(0.4))
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
    }
}

/// Reused non-diagnostic disclaimer.
struct DisclaimerNote: View {
    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "info.circle").foregroundStyle(Theme.inkSoft).font(.caption)
            Text("This is a technology demo, not a medical or diagnostic assessment.")
                .font(.caption).foregroundStyle(Theme.inkSoft)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
