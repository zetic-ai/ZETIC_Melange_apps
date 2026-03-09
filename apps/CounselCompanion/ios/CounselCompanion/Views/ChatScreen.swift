import SwiftUI

struct ChatScreen: View {
    @EnvironmentObject var vm: ChatViewModel

    var body: some View {
        VStack(spacing: 0) {
            header
                .padding(.horizontal, 16)
                .padding(.top, 8)
                .padding(.bottom, 8)

            if vm.isDownloading {
                downloadProgressView
                    .padding(.horizontal, 16)
                    .padding(.bottom, 6)
            } else if vm.isGenerating || vm.isModelLoading {
                statusPill
                    .padding(.bottom, 6)
            }

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 10) {
                        ForEach(vm.messages(for: vm.currentSessionID ?? UUID())) { msg in
                            bubble(for: msg)
                                .id(msg.id)
                        }
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 6)
                }
                .scrollDismissesKeyboard(.interactively)
                .onChange(of: vm.messages(for: vm.currentSessionID ?? UUID()).count) { _ in
                    scrollToBottom(proxy)
                }
                .onChange(of: vm.scrollTrigger) { _ in
                    scrollToBottom(proxy)
                }
            }

            inputBar
                .padding(.horizontal, 14)
                .padding(.top, 6)
                .padding(.bottom, 4)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(
            LinearGradient(
                colors: [Color.bgTop, Color.bgBottom],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea(.container)
        )
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Counsel Companion")
                    .font(.system(size: 17, weight: .semibold, design: .serif))
                Text(vm.sessions.first(where: { $0.id == vm.currentSessionID })?.title ?? "New conversation")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button {
                vm.createSession()
            } label: {
                Image(systemName: "plus")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(Color.warmSage)
                    .frame(width: 32, height: 32)
                    .background(Color.warmSage.opacity(0.12), in: Circle())
            }
        }
    }

    private var downloadProgressView: some View {
        VStack(spacing: 8) {
            Text(vm.initializationState)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(Color.warmSage)

            if vm.downloadProgress > 0.0 && vm.downloadProgress < 1.0 {
                ProgressView(value: vm.downloadProgress)
                    .progressViewStyle(LinearProgressViewStyle(tint: Color.warmSage))
            } else {
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle())
                    .tint(Color.warmSage)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity)
        .background(Color.warmSage.opacity(0.08), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private var statusPill: some View {
        HStack(spacing: 6) {
            ProgressView()
                .controlSize(.mini)
                .tint(Color.warmSage)
            Text(vm.loadingMessage)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Color.warmSage)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 5)
        .background(Color.warmSage.opacity(0.1), in: Capsule())
    }

    private func bubble(for message: Message) -> some View {
        let isUser = message.role == .user

        return HStack {
            if isUser { Spacer(minLength: 60) }
            Text(message.content.isEmpty && vm.isGenerating && !isUser ? "Thinking..." : message.content)
                .font(.system(size: 14))
                .lineSpacing(3)
                .padding(.horizontal, 12)
                .padding(.vertical, 9)
                .background(
                    isUser
                        ? AnyShapeStyle(LinearGradient(
                            colors: [Color.warmSage, Color.warmSage.opacity(0.85)],
                            startPoint: .topLeading, endPoint: .bottomTrailing))
                        : AnyShapeStyle(Color(.tertiarySystemBackground))
                )
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                .foregroundStyle(isUser ? .white : .primary)
            if !isUser { Spacer(minLength: 60) }
        }
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        if let last = vm.messages(for: vm.currentSessionID ?? UUID()).last {
            withAnimation(.easeOut(duration: 0.3)) {
                proxy.scrollTo(last.id, anchor: .bottom)
            }
        }
    }

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField("Share how you feel...", text: $vm.draft, axis: .vertical)
                .font(.system(size: 15, weight: .regular))
                .lineLimit(4)
                .padding(.horizontal, 14)
                .padding(.vertical, 11)
                .background(Color(.tertiarySystemBackground), in: RoundedRectangle(cornerRadius: 22, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .stroke(Color.warmSage.opacity(0.35), lineWidth: 1)
                )
                .disabled(vm.isGenerating || vm.isModelLoading || vm.isDownloading)

            Button {
                if vm.isGenerating { vm.stopGeneration() } else { vm.sendMessage() }
            } label: {
                Image(systemName: vm.isGenerating ? "stop.fill" : "arrow.up")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 34, height: 34)
                    .background(
                        vm.isGenerating ? Color.warmPeach : Color.warmSage,
                        in: Circle()
                    )
            }
            .disabled(vm.isModelLoading || vm.isDownloading || (!vm.isGenerating && vm.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty))
            .opacity(vm.isModelLoading || vm.isDownloading || (!vm.isGenerating && vm.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) ? 0.4 : 1)
        }
    }
}
