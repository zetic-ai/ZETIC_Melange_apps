import SwiftUI

struct SettingsScreen: View {
    @EnvironmentObject var vm: ChatViewModel
    @State private var promptDraft: String = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Settings")
                    .font(.system(size: 17, weight: .semibold, design: .serif))
                    .padding(.horizontal, 16)
                    .padding(.top, 8)

                settingsCard {
                    Text("Appearance")
                        .font(.system(size: 13, weight: .medium))

                    HStack(spacing: 8) {
                        themeChip("System", icon: "iphone", mode: nil)
                        themeChip("Light", icon: "sun.max", mode: .light)
                        themeChip("Dark", icon: "moon", mode: .dark)
                    }
                }

                settingsCard {
                    Text("System Prompt")
                        .font(.system(size: 13, weight: .medium))

                    TextEditor(text: $promptDraft)
                        .font(.system(size: 13))
                        .frame(minHeight: 90)
                        .scrollContentBackground(.hidden)
                        .padding(10)
                        .background(
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .fill(Color(.quaternarySystemFill))
                        )

                    Button {
                        vm.setSystemPrompt(promptDraft.trimmingCharacters(in: .whitespacesAndNewlines))
                    } label: {
                        Text("Save Prompt")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(Color.warmSage, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    }
                }

                settingsCard {
                    Text("Data")
                        .font(.system(size: 13, weight: .medium))

                    Button(role: .destructive) {
                        vm.clearHistory()
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "trash")
                                .font(.system(size: 11))
                            Text("Clear All History")
                                .font(.system(size: 13))
                        }
                        .foregroundStyle(Color.warmPeach)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .background(Color.warmPeach.opacity(0.1), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    }
                }
            }
            .padding(.bottom, 24)
        }
        .onAppear { promptDraft = vm.systemPrompt }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(
            LinearGradient(colors: [Color.bgTop, Color.bgBottom], startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea(.container)
        )
    }

    private func settingsCard(@ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 10) {
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

    private func themeChip(_ label: String, icon: String, mode: ColorScheme?) -> some View {
        let isSelected = (mode == nil && vm.themeOverride == nil) || vm.themeOverride == mode

        return Button { vm.setTheme(mode) } label: {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 14))
                Text(label)
                    .font(.system(size: 10, weight: .medium))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(isSelected ? Color.warmSage.opacity(0.15) : Color(.quaternarySystemFill))
            )
            .foregroundStyle(isSelected ? Color.warmSage : Color.secondary)
        }
        .buttonStyle(.plain)
    }
}
