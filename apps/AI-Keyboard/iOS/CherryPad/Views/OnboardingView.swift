import SwiftUI

/// First-launch guide to enable the CherryPad keyboard and grant Full Access. The
/// full AI experience lives in this app; the keyboard is a companion that hands
/// off into it.
struct OnboardingView: View {
    var onDone: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 8) {
                Text("🍒").font(.system(size: 44))
                Text("CherryPad")
                    .font(.system(size: 30, weight: .heavy, design: .rounded))
                    .foregroundStyle(Theme.textPrimary)
                Text("On-device AI for rewriting, replying, translating, and fixing grammar — all private to your iPhone.")
                    .font(.system(size: 16))
                    .foregroundStyle(Theme.textSecondary)
            }
            .padding(.top, 24)
            .padding(.bottom, 28)

            VStack(alignment: .leading, spacing: 18) {
                step(1, "Add the keyboard", "Settings ▸ General ▸ Keyboard ▸ Keyboards ▸ Add New Keyboard ▸ CherryPad.")
                step(2, "Allow Full Access", "Tap CherryPad in the keyboard list and turn on Allow Full Access — needed to hand your text to this app. No keystrokes ever leave your device.")
                step(3, "Use it anywhere", "Switch to CherryPad with 🌐, select text, and tap an action. We'll bring you here to generate, then you can paste the result back.")
            }

            Spacer()

            Button {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            } label: {
                Text("Open Settings")
                    .font(.system(size: 16, weight: .bold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 15)
                    .foregroundStyle(Theme.onCherry)
                    .background(RoundedRectangle(cornerRadius: Theme.chipRadius, style: .continuous).fill(Theme.cherry))
            }
            .buttonStyle(.plain)

            Button(action: onDone) {
                Text("Start using CherryPad")
                    .font(.system(size: 15, weight: .semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .foregroundStyle(Theme.cherry)
            }
            .buttonStyle(.plain)
            .padding(.bottom, 8)
        }
        .padding(.horizontal, 24)
        .background(Theme.background.ignoresSafeArea())
    }

    private func step(_ n: Int, _ title: String, _ detail: String) -> some View {
        HStack(alignment: .top, spacing: 14) {
            Text("\(n)")
                .font(.system(size: 15, weight: .bold))
                .foregroundStyle(Theme.onCherry)
                .frame(width: 28, height: 28)
                .background(Circle().fill(Theme.cherry))
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.system(size: 16, weight: .semibold)).foregroundStyle(Theme.textPrimary)
                Text(detail).font(.system(size: 14)).foregroundStyle(Theme.textSecondary)
            }
        }
    }
}
