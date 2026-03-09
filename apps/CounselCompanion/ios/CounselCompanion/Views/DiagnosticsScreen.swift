import SwiftUI

struct DiagnosticsScreen: View {
    @EnvironmentObject var vm: ChatViewModel
    private let formatter: DateFormatter = {
        let f = DateFormatter(); f.dateStyle = .medium; f.timeStyle = .medium; return f
    }()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text("Diagnostics")
                    .font(.system(size: 17, weight: .semibold, design: .serif))
                    .padding(.horizontal, 16)
                    .padding(.top, 8)

                diagCard("Model Info") {
                    infoRow("Model ID", ZeticChatEngine.modelId)
                    Divider()
                    infoRow("Personal Key", ZeticChatEngine.maskedKey())
                    Divider()
                    infoRow("Key Status", ZeticChatEngine.maskedKey().count > 8 ? "Configured" : "Missing")
                }

                diagCard("Last Generation") {
                    infoRow("Timestamp", vm.diagnostics.lastRun.map { formatter.string(from: $0) } ?? "No run yet")
                    Divider()
                    infoRow("Duration", "\(vm.diagnostics.lastDurationMs) ms")
                    Divider()
                    infoRow("Tokens", "\(vm.diagnostics.lastTokenCount)")
                    Divider()
                    infoRow("Stop Reason", vm.diagnostics.lastStopReason)
                }

                diagCard("Raw Log") {
                    Text(vm.diagnostics.lastRawLog.isEmpty ? "No generation yet." : vm.diagnostics.lastRawLog)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
            .padding(.bottom, 24)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(
            LinearGradient(colors: [Color.bgTop, Color.bgBottom], startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea(.container)
        )
    }

    private func diagCard(_ title: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.system(size: 13, weight: .medium))
            content()
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(.tertiarySystemBackground))
        )
        .padding(.horizontal, 14)
    }

    private func infoRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.system(size: 11, weight: .medium))
        }
    }
}
