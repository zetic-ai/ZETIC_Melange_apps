import SwiftUI

struct SessionsScreen: View {
    @EnvironmentObject var vm: ChatViewModel
    private let formatter: DateFormatter = {
        let f = DateFormatter(); f.dateStyle = .medium; f.timeStyle = .short; return f
    }()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Your Sessions")
                        .font(.system(size: 17, weight: .semibold, design: .serif))
                    Text("\(vm.sessions.count) conversations")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 16)
                .padding(.top, 8)

                LazyVStack(spacing: 8) {
                    ForEach(vm.sessions) { session in
                        let selected = session.id == vm.currentSessionID
                        Button { vm.selectSession(session.id) } label: {
                            HStack(alignment: .top, spacing: 10) {
                                Image(systemName: "bubble.left")
                                    .font(.system(size: 12))
                                    .foregroundStyle(selected ? Color.warmSage : Color.secondary)
                                    .frame(width: 20)
                                    .padding(.top, 2)

                                VStack(alignment: .leading, spacing: 3) {
                                    HStack {
                                        Text(session.title)
                                            .font(.system(size: 13, weight: .medium))
                                            .foregroundStyle(.primary)
                                            .lineLimit(1)
                                        Spacer()
                                        if selected {
                                            Text("Active")
                                                .font(.system(size: 9, weight: .medium))
                                                .foregroundStyle(Color.warmSage)
                                                .padding(.horizontal, 6)
                                                .padding(.vertical, 2)
                                                .background(Color.warmSage.opacity(0.12), in: Capsule())
                                        }
                                    }
                                    Text(formatter.string(from: session.updatedAt))
                                        .font(.system(size: 10))
                                        .foregroundStyle(.secondary)
                                    Text(session.messages.last?.content ?? "No messages yet")
                                        .font(.system(size: 11))
                                        .foregroundStyle(.secondary)
                                        .lineLimit(2)
                                }
                            }
                            .padding(12)
                            .background(
                                RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    .fill(selected
                                          ? Color.warmSage.opacity(0.08)
                                          : Color(.tertiarySystemBackground))
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    .strokeBorder(selected ? Color.warmSage.opacity(0.2) : .clear, lineWidth: 1)
                            )
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 14)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(
            LinearGradient(colors: [Color.bgTop, Color.bgBottom], startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea(.container)
        )
    }
}
