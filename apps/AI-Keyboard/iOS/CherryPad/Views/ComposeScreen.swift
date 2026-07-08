import SwiftUI

/// The main MangoPad-style compose surface: header, input card, action bar,
/// context chips, Generate, and the streamed result card.
struct ComposeScreen: View {
    @ObservedObject var model: AppModel
    @ObservedObject var llm: LLMService
    @Binding var showSettings: Bool

    @State private var showLanguagePicker = false
    @FocusState private var inputFocused: Bool

    var body: some View {
        ZStack(alignment: .top) {
            Theme.background.ignoresSafeArea()

            ScrollView {
                VStack(spacing: 18) {
                    header

                    ModelStatusBanner(llm: llm)

                    inputCard

                    ActionBar(selection: $model.task)

                    if model.task != .grammar {
                        ChipRow(
                            task: model.task,
                            tone: $model.tone,
                            stance: $model.stance,
                            targetLanguage: $model.targetLanguage,
                            onPickLanguage: { showLanguagePicker = true }
                        )
                    }

                    generateButton

                    if model.hasResult {
                        ResultCard(model: model)
                            .transition(.move(edge: .bottom).combined(with: .opacity))
                    }
                }
                .padding(.horizontal, 18)
                .padding(.bottom, 40)
            }
            .scrollDismissesKeyboard(.interactively)
        }
        .animation(.snappy(duration: 0.22), value: model.hasResult)
        .animation(.snappy(duration: 0.18), value: model.task)
        .sheet(isPresented: $showLanguagePicker) {
            LanguagePickerView(selected: model.targetLanguage) { model.targetLanguage = $0 }
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "applelogo").opacity(0) // spacer balance
                .frame(width: 28)
            Spacer()
            HStack(spacing: 7) {
                Text("🍒").font(.system(size: 22))
                Text("CherryPad")
                    .font(.system(size: 22, weight: .heavy, design: .rounded))
                    .foregroundStyle(Theme.textPrimary)
            }
            Spacer()
            Button { showSettings = true } label: {
                Image(systemName: "gearshape.fill")
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(Theme.textSecondary)
                    .frame(width: 28)
            }
        }
        .padding(.top, 8)
    }

    private var inputCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            ZStack(alignment: .topLeading) {
                if model.inputText.isEmpty {
                    Text("Type or paste your message…")
                        .font(.system(size: 16))
                        .foregroundStyle(Theme.textSecondary)
                        .padding(.top, 8)
                        .padding(.leading, 5)
                }
                TextEditor(text: $model.inputText)
                    .font(.system(size: 16))
                    .foregroundStyle(Theme.textPrimary)
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 120, maxHeight: 200)
                    .focused($inputFocused)
            }
            HStack {
                Spacer()
                Text("\(model.inputText.count)")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.textSecondary)
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous)
                .fill(Theme.surface)
                .shadow(color: Theme.cardShadow, radius: 10, y: 3)
        )
    }

    private var generateButton: some View {
        Button {
            inputFocused = false
            model.run()
        } label: {
            HStack(spacing: 7) {
                Image(systemName: "sparkles")
                Text(model.task.title)
            }
            .font(.system(size: 16, weight: .bold))
            .frame(maxWidth: .infinity)
            .padding(.vertical, 15)
            .foregroundStyle(Theme.onCherry)
            .background(
                RoundedRectangle(cornerRadius: Theme.chipRadius, style: .continuous)
                    .fill(model.canGenerate ? Theme.cherry : Theme.cherry.opacity(0.4))
            )
        }
        .buttonStyle(.plain)
        .disabled(!model.canGenerate)
    }
}
