import SwiftUI
import UIKit

/// Single state-driven screen mirroring DeepL's flow: idle → editing → result, with a
/// persistent language bar. Matches reference screenshots #4 (idle), #2 (editing), #3 (result).
struct TranslatorScreen: View {
    @StateObject private var vm = TranslationViewModel()
    @StateObject private var network = NetworkMonitor()
    @State private var phase: Phase = .idle
    @State private var picker: PickerField?
    @State private var showImageChooser = false
    @State private var pickerSource: ImagePicker.Source?
    @FocusState private var inputFocused: Bool

    enum Phase { case idle, editing, result }
    enum PickerField: Identifiable {
        case source, target
        var id: Int { self == .source ? 0 : 1 }
    }

    var body: some View {
        ZStack {
            Theme.background.ignoresSafeArea()

            VStack(spacing: 11) {
                header.padding(.horizontal, 17)
                card.padding(.horizontal, 14)
                LanguageBar(
                    source: vm.source,
                    target: vm.target,
                    onTapSource: { picker = .source },
                    onTapTarget: { picker = .target },
                    onSwap: { withAnimation(.easeInOut) { vm.swapLanguages() } }
                )
                .padding(.horizontal, 14)
                PoweredByZetic().padding(.bottom, 4)
            }
            .padding(.top, 8)
        }
        .preferredColorScheme(.dark)
        .onAppear { applySeedIfPresent() }
        .sheet(item: $picker) { field in
            switch field {
            case .source:
                LanguagePickerView(title: "Translate from", options: Language.sourceOptions, selected: vm.source) {
                    vm.source = $0; retranslateIfNeeded()
                }
            case .target:
                LanguagePickerView(title: "Translate to", options: Language.targetOptions, selected: vm.target) {
                    vm.target = $0; retranslateIfNeeded()
                }
            }
        }
        // Voice/OCR captured text → present the result screen (translation auto-starts in the VM).
        .onChange(of: vm.showResultSignal) { _ in
            inputFocused = false
            withAnimation(.easeInOut) { phase = .result }
        }
        .confirmationDialog("Translate from image", isPresented: $showImageChooser, titleVisibility: .visible) {
            if UIImagePickerController.isSourceTypeAvailable(.camera) {
                Button("Take Photo") { pickerSource = .camera }
            }
            Button("Choose from Library") { pickerSource = .library }
            Button("Cancel", role: .cancel) {}
        }
        .sheet(item: $pickerSource) { source in
            ImagePicker(source: source) { image in
                if let image { vm.recognize(image: image) }
            }
            .ignoresSafeArea()
        }
        .alert("Input failed", isPresented: inputErrorBinding) {
            Button("OK", role: .cancel) { vm.inputError = nil }
        } message: {
            Text(vm.inputError ?? "")
        }
    }

    private var inputErrorBinding: Binding<Bool> {
        Binding(get: { vm.inputError != nil }, set: { if !$0 { vm.inputError = nil } })
    }

    // MARK: - Header

    @ViewBuilder private var header: some View {
        switch phase {
        case .idle:    idleHeader
        case .editing: editingHeader
        case .result:  resultHeader
        }
    }

    private var idleHeader: some View {
        ZStack {
            segmentedControl.fixedSize()
            HStack {
                Image(systemName: "person.crop.circle.fill")
                    .font(.system(size: 30))
                    .foregroundStyle(Theme.accent)
                Spacer()
                LiveStatusBadge(isOnline: network.isOnline)
            }
        }
        .frame(height: 40)
    }

    private var editingHeader: some View {
        HStack {
            backButton { withAnimation { phase = .idle }; inputFocused = false }
            Spacer()
            LiveStatusBadge(isOnline: network.isOnline)
            Spacer()
            Button(action: confirmTranslate) {
                Image(systemName: "checkmark")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(vm.canTranslate ? Theme.textPrimary : Theme.textTertiary)
            }
            .disabled(!vm.canTranslate)
        }
        .frame(height: 40)
    }

    private var resultHeader: some View {
        HStack {
            backButton { withAnimation { phase = .editing }; inputFocused = true }
            Spacer()
            LiveStatusBadge(isOnline: network.isOnline)
            Spacer()
            Button { vm.clearAll(); withAnimation { phase = .idle } } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(Theme.textPrimary)
            }
        }
        .frame(height: 40)
    }

    private var segmentedControl: some View {
        HStack(spacing: 4) {
            segment("Translator", icon: "character.bubble", selected: true)
            segment("", icon: "wand.and.stars", selected: false)
        }
        .padding(4)
        .background(Theme.surfaceRaised, in: Capsule())
    }

    private func segment(_ title: String, icon: String, selected: Bool) -> some View {
        HStack(spacing: 5) {
            Image(systemName: icon).font(.system(size: 12, weight: .semibold))
            Text(title).font(.system(size: 14, weight: .semibold))
        }
        .foregroundStyle(selected ? .white : Theme.textSecondary)
        .padding(.horizontal, 13)
        .padding(.vertical, 6)
        .background(selected ? Theme.accentDeep : .clear, in: Capsule())
    }

    // MARK: - Centered model-loading overlay

    @ViewBuilder private var modelLoadingOverlay: some View {
        switch vm.modelState {
        case .ready:
            EmptyView()
        case .loading:
            loadingCard {
                ProgressView()
                    .controlSize(.large)
                    .tint(Theme.accent)
                Text(loadingTitle)
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(Theme.textPrimary)
                if vm.downloadProgress > 0 && vm.downloadProgress < 1 {
                    ProgressView(value: vm.downloadProgress)
                        .tint(Theme.accent)
                        .frame(maxWidth: 220)
                }
                Text("Preparing on-device translation. The model downloads once, then works fully offline.")
                    .font(.system(size: 13))
                    .foregroundStyle(Theme.textSecondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
        case .failed(let message):
            loadingCard {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 26))
                    .foregroundStyle(.orange)
                Text("Couldn't load the model")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(Theme.textPrimary)
                Text(message)
                    .font(.system(size: 13))
                    .foregroundStyle(Theme.textSecondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                Button { vm.loadModel() } label: {
                    Text("Retry")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 24).padding(.vertical, 10)
                        .background(Theme.accent, in: Capsule())
                }
                .padding(.top, 4)
            }
        }
    }

    private var loadingTitle: String {
        (vm.downloadProgress > 0 && vm.downloadProgress < 1)
            ? "Downloading model… \(Int(vm.downloadProgress * 100))%"
            : "Preparing model…"
    }

    private func loadingCard<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        ZStack {
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(Theme.surface)
            VStack(spacing: 14, content: content)
                .padding(28)
        }
    }

    private func backButton(_ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: "arrow.left")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(Theme.textPrimary)
        }
    }

    // MARK: - Card

    private var card: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 22, style: .continuous).fill(Theme.surface)
            Group {
                switch phase {
                case .idle:    IdleView(vm: vm, onActivate: startEditing, onVoice: startVoice, onImage: { showImageChooser = true })
                case .editing: EditingView(vm: vm, focused: $inputFocused)
                case .result:  ResultView(vm: vm, onEditSource: startEditing)
                }
            }
            .padding(20)
        }
        .overlay { modelLoadingOverlay }
        .overlay { inputOverlay }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Voice / OCR overlay

    @ViewBuilder private var inputOverlay: some View {
        if vm.isListening {
            inputCard {
                Image(systemName: "mic.fill")
                    .font(.system(size: 40))
                    .foregroundStyle(Theme.accent)
                Text("Listening…")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(Theme.textPrimary)
                Text(vm.partialVoiceText.isEmpty ? "Speak now, then tap Stop." : vm.partialVoiceText)
                    .font(.system(size: vm.partialVoiceText.isEmpty ? 13 : 18))
                    .foregroundStyle(vm.partialVoiceText.isEmpty ? Theme.textSecondary : Theme.textPrimary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                Button(action: { vm.stopVoiceInput() }) {
                    Text("Stop")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 24).padding(.vertical, 10)
                        .background(Theme.accent, in: Capsule())
                }
                .padding(.top, 4)
            }
        } else if vm.isRecognizingImage {
            inputCard {
                ProgressView().controlSize(.large).tint(Theme.accent)
                Text("Reading text from image…")
                    .font(.system(size: 15))
                    .foregroundStyle(Theme.textSecondary)
            }
        }
    }

    private func inputCard<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        ZStack {
            RoundedRectangle(cornerRadius: 22, style: .continuous).fill(Theme.surface)
            VStack(spacing: 14, content: content).padding(28)
        }
    }

    private func startVoice() {
        inputFocused = false
        vm.startVoiceInput()
    }

    // MARK: - Actions

    private func startEditing() {
        withAnimation { phase = .editing }
        inputFocused = true
    }

    private func confirmTranslate() {
        inputFocused = false
        vm.translate()
        withAnimation { phase = .result }
    }

    private func retranslateIfNeeded() {
        if phase == .result { vm.translate() }
    }

    /// Lets `xcrun simctl ... TG_SEED=editing|result` jump straight to a state for
    /// screenshots/demo. No-op unless the env var is set (preview/mock target only).
    private func applySeedIfPresent() {
        guard let seed = ProcessInfo.processInfo.environment["TG_SEED"] else { return }
        vm.source = .named("ko")
        vm.target = .named("en")
        vm.sourceText = "Zetic을 사용하여 어떤 기기에서든 로컬로 나만의 AI 모델을 배포하세요."
        switch seed {
        case "editing":
            phase = .editing
        case "result":
            phase = .result
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) { vm.translate() }
        default:
            break
        }
    }
}
