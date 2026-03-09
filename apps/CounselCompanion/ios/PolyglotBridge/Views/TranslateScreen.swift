import SwiftUI
import UIKit

struct TranslateScreen: View {
    @EnvironmentObject private var app: AppViewModel
    @State private var draft: String = ""
    @State private var source: Language
    @State private var target: Language
    @State private var scrollID = UUID()
    @State private var showCopiedToast = false

    init() {
        let settings = SettingsStore().load()
        _source = State(initialValue: Language.byCode(settings.defaultSource))
        _target = State(initialValue: Language.byCode(settings.defaultTarget))
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            if !app.modelReady { prepareCard }
            messageList
            inputBar
        }
        .background(Color(.systemGroupedBackground))
        .onReceive(NotificationCenter.default.publisher(for: .useFavorite)) { notif in
            if let phrase = notif.object as? FavoritePhrase {
                draft = phrase.text
                source = phrase.source
                target = phrase.target
            }
        }
        .onTapGesture { hideKeyboard() }
        .animation(.easeInOut, value: app.isGenerating)
    }

    private var header: some View {
        HStack(spacing: 12) {
            languagePicker(title: "From", selection: $source)
            Button(action: swapLanguages) {
                Image(systemName: "arrow.right.arrow.left.circle.fill")
                    .font(.system(size: 24, weight: .semibold))
                    .foregroundStyle(AppColors.accentGradient)
            }
            languagePicker(title: "To", selection: $target)
        }
        .padding()
        .background(AppColors.accent.opacity(0.07))
    }

    private func languagePicker(title: String, selection: Binding<Language>) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundColor(.secondary)
            Menu {
                ForEach(app.languages) { lang in
                    Button(action: { selection.wrappedValue = lang }) {
                        Text("\(lang.flag) \(lang.name)")
                    }
                }
            } label: {
                HStack {
                    Text("\(selection.wrappedValue.flag) \(selection.wrappedValue.name)")
                        .font(.headline)
                    Image(systemName: "chevron.down")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
    }

    private var prepareCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Download / Prepare Model")
                .font(AppFont.title(20))
            Text("We need to load the HY-MT model before first translation. This happens locally.")
                .font(AppFont.body())
                .foregroundColor(.secondary)
            ProgressView(value: app.preparationProgress, total: 1.0)
                .progressViewStyle(.linear)
            Button(action: { Task { await app.prepareModel() } }) {
                HStack {
                    if app.preparationProgress > 0 && app.preparationProgress < 1 {
                        ProgressView()
                    }
                    Text("Prepare now")
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
        .background(GlassBackground())
        .padding(.horizontal)
    }

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(app.messages(for: app.currentSessionID ?? UUID())) { message in
                        MessageBubble(message: message)
                            .id(message.id)
                    }
                }
                .padding(.horizontal)
                .padding(.top, 8)
            }
            .onChange(of: app.messages(for: app.currentSessionID ?? UUID()).last?.id) { id in
                if let id = id { withAnimation { proxy.scrollTo(id, anchor: .bottom) } }
            }
        }
    }

    private var inputBar: some View {
        VStack(spacing: 10) {
            if let error = app.errorMessage {
                Text(error)
                    .foregroundColor(.red)
                    .font(.footnote)
            }
            HStack(alignment: .bottom, spacing: 12) {
                TextEditor(text: $draft)
                    .frame(minHeight: 48, maxHeight: 140)
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 12).fill(Color(.secondarySystemBackground)))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .strokeBorder(app.isGenerating ? AppColors.accent : Color(.separator), lineWidth: 1)
                    )
                VStack(spacing: 8) {
                    Button(action: send) {
                        Image(systemName: "paperplane.fill")
                            .foregroundStyle(Color.white)
                            .frame(width: 44, height: 44)
                            .background(Circle().fill(AppColors.accentGradient))
                    }
                    Button(action: stop) {
                        Image(systemName: "stop.fill")
                            .foregroundColor(.white)
                            .frame(width: 44, height: 44)
                            .background(Circle().fill(Color.red.opacity(0.9)))
                    }
                    .opacity(app.isGenerating ? 1 : 0.35)
                    .disabled(!app.isGenerating)
                }
            }
            .padding(.horizontal)
            HStack {
                if app.isGenerating {
                    ProgressView()
                }
                Text(app.loadingMessage)
                    .font(.footnote)
                    .foregroundColor(.secondary)
                Spacer()
                Button(action: copyLast) {
                    Image(systemName: "doc.on.doc")
                }
                .disabled(app.messages(for: app.currentSessionID ?? UUID()).last?.role != .assistant)
                if showCopiedToast {
                    Text("Copied")
                        .font(.footnote)
                        .foregroundColor(.green)
                        .transition(.opacity)
                }
            }
            .padding(.horizontal)
            .padding(.bottom, 8)
        }
        .padding(.top, 4)
        .background(VisualEffectBlur())
    }

    private func send() {
        guard app.modelReady else { return }
        app.send(text: draft, source: source, target: target)
        draft = ""
    }

    private func stop() {
        app.stopGeneration()
    }

    private func swapLanguages() {
        let temp = source
        source = target
        target = temp
    }

    private func copyLast() {
        guard let sessionID = app.currentSessionID,
              let last = app.messages(for: sessionID).last(where: { $0.role == .assistant }) else { return }
        UIPasteboard.general.string = last.content
        withAnimation { showCopiedToast = true }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            withAnimation { showCopiedToast = false }
        }
    }
}

private struct MessageBubble: View {
    @EnvironmentObject private var app: AppViewModel
    let message: Message

    var body: some View {
        HStack {
            if message.role == .assistant {
                Spacer()
            }
            VStack(alignment: .leading, spacing: 6) {
                Text(message.role == .user ? "You" : "HY-MT")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(message.content.isEmpty ? "…" : message.content)
                    .font(AppFont.body())
                    .foregroundColor(message.role == .user ? .primary : .white)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(12)
            .background(message.role == .user ? Color(.secondarySystemBackground) : AppColors.accentGradient)
            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay(alignment: .topTrailing) {
                if message.role == .assistant {
                    Button(action: saveFavorite) {
                        Image(systemName: "star.fill")
                            .font(.caption)
                            .foregroundColor(.yellow)
                            .padding(6)
                    }
                }
            }
            if message.role == .user {
                Spacer()
            }
        }
    }

    private func saveFavorite() {
        let sourceLang = Language.byCode(message.source)
        let targetLang = Language.byCode(message.target)
        app.saveFavorite(from: message, source: sourceLang, target: targetLang)
    }
}

struct VisualEffectBlur: UIViewRepresentable {
    func makeUIView(context: Context) -> UIVisualEffectView { UIVisualEffectView(effect: UIBlurEffect(style: .systemChromeMaterial)) }
    func updateUIView(_ uiView: UIVisualEffectView, context: Context) {}
}

extension View {
    func hideKeyboard() {
        #if canImport(UIKit)
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
        #endif
    }
}
